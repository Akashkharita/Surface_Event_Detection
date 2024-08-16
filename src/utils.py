import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import obspy
from obspy.signal.filter import envelope
from obspy.clients.fdsn import Client
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, RepeatedKFold
from sklearn.metrics import (accuracy_score, roc_curve, roc_auc_score, auc, 
                             classification_report, confusion_matrix, f1_score, 
                             precision_score, recall_score)
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats, signal
from sklearn.datasets import load_iris
from obspy.geodetics.base import gps2dist_azimuth
from datetime import datetime, timedelta
import time
import re
import tsfel
import random
import calendar
import concurrent.futures
import seaborn as sns
from scipy.signal import resample
import matplotlib.lines as mlines
from joblib import dump, load
from math import radians, sin, cos, sqrt, atan2

# Custom imports (ensure path to the seis_feature library is correct)
#import sys
#sys.path.append('../Feature_Extraction_Scripts/Physical_Feature_Extraction_Scripts/')
#import seis_feature
#from seis_feature import compute_physical_features

pd.set_option('display.max_columns', None)

# TSFEL configuration for feature extraction
cfg_file = tsfel.get_features_by_domain()
from tsfel import time_series_features_extractor


def resample_array(arr, original_rate, desired_rate):
    """
    Resample a given array to a new sampling rate using scipy's resample function.

    Parameters:
        arr (numpy array): Array to be resampled.
        original_rate (float): Original sampling rate in Hz.
        desired_rate (float): Desired sampling rate in Hz.

    Returns:
        numpy array: Resampled array.
    """
    num_samples = len(arr)
    duration = num_samples / original_rate  # Duration of the array in seconds
    new_num_samples = int(duration * desired_rate)
    return resample(arr, new_num_samples)


def apply_cosine_taper(arrays, taper_percent=10):
    """
    Apply a cosine taper to each array in an array of arrays.

    Parameters:
        arrays (numpy array): 2D array where each sub-array is tapered.
        taper_percent (float): Percentage of the array to taper at both ends.

    Returns:
        numpy array: Tapered arrays.
    """
    tapered_arrays = []
    num_samples = arrays.shape[1]  # Assuming each sub-array has the same length

    for array in arrays:
        taper_length = int(num_samples * taper_percent / 100)
        taper_window = np.hanning(2 * taper_length)

        tapered_array = array.copy()
        tapered_array[:taper_length] *= taper_window[:taper_length]
        tapered_array[-taper_length:] *= taper_window[taper_length:]

        tapered_arrays.append(tapered_array)

    return np.array(tapered_arrays)


def butterworth_filter(arrays, lowcut=1, highcut=10, fs=100, num_corners=4, filter_type='bandpass'):
    """
    Apply a Butterworth filter (bandpass, highpass, or lowpass) to each array in a list of arrays using filtfilt.

    Parameters:
        arrays (list of numpy arrays): List of arrays to be filtered.
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        num_corners (int): Number of corners (filter order).
        filter_type (str, optional): Type of filter ('bandpass', 'highpass', or 'lowpass'). Default is 'bandpass'.

    Returns:
        list of numpy arrays: List of filtered arrays.
    """
    filtered_arrays = []
    for data in arrays:
        # Normalize the frequency values to Nyquist frequency (0.5 * fs)
        lowcut_norm = lowcut / (0.5 * fs)
        highcut_norm = highcut / (0.5 * fs)

        # Design the Butterworth filter based on the filter type
        if filter_type == 'bandpass':
            b, a = signal.butter(num_corners, [lowcut_norm, highcut_norm], btype='band')
        elif filter_type == 'highpass':
            b, a = signal.butter(num_corners, lowcut_norm, btype='high')
        elif filter_type == 'lowpass':
            b, a = signal.butter(num_corners, highcut_norm, btype='low')
        else:
            raise ValueError("Invalid filter_type. Use 'bandpass', 'highpass', or 'lowpass'.")

        # Apply the filter to the data using filtfilt
        filtered_data = signal.filtfilt(b, a, data)

        filtered_arrays.append(filtered_data)

    return filtered_arrays


def plot_confusion_matrix(cf, class_labels=None, figure_name='confusion_matrix.png'):
    """
    Plot a confusion matrix heatmap.

    Parameters:
        cf (array-like): Confusion matrix to be plotted.
        class_labels (list, optional): Labels for the classes. Default is ['Earthquake', 'Explosion', 'Noise', 'Surface'].
        figure_name (str, optional): Name of the file to save the plot. Default is 'confusion_matrix.png'.
    """
    if class_labels is None:
        class_labels = ['Earthquake', 'Explosion', 'Noise', 'Surface']

    plt.figure(figsize=[8, 6])
    annot_kws = {"fontsize": 15}

    # Plot the heatmap
    ax = sns.heatmap(cf, annot=True, cmap='Blues', fmt='d', 
                     xticklabels=class_labels, yticklabels=class_labels, 
                     annot_kws=annot_kws)

    # Set labels and title
    ax.set_xticklabels(class_labels, fontsize=15)
    ax.set_yticklabels(class_labels, fontsize=15)
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('Actual', fontsize=15)
    plt.tight_layout()
    plt.savefig(figure_name)


def plot_classification_report(cr, class_labels=None, figure_name='classification_report.png'):
    """
    Plot a classification report heatmap.

    Parameters:
        cr (array-like): Classification report data to be plotted.
        class_labels (list, optional): Labels for the classes. Default is ['Earthquake', 'Explosion', 'Noise', 'Surface'].
        figure_name (str, optional): Name of the file to save the plot. Default is 'classification_report.png'.
    """
    if class_labels is None:
        class_labels = ['Earthquake', 'Explosion', 'Noise', 'Surface']

    labels = ['Precision', 'Recall', 'F1-Score']
    plt.figure(figsize=[8, 6])
    annot_kws = {"fontsize": 15}

    # Plot the heatmap
    ax = sns.heatmap(pd.DataFrame(cr).iloc[:3, :len(class_labels)], annot=True, 
                     cmap='Blues', yticklabels=labels, xticklabels=class_labels, 
                     vmin=0.8, vmax=1, annot_kws=annot_kws)

    # Set labels and title
    ax.set_xticklabels(class_labels, fontsize=15)
    ax.set_yticklabels(labels, fontsize=15)
    ax.set_xlabel('Metrics', fontsize=15)
    ax.set_ylabel('Classes', fontsize=15)
    ax.set_title('Classification Report', fontsize=18)
    plt.tight_layout()
    plt.savefig(figure_name)


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface using the Haversine formula.

    Parameters:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

    Returns:
        float: Distance between the two points in kilometers.
    """
    # Convert latitude and longitude from degrees to radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    # Radius of the Earth in kilometers
    R = 6371.0

    # Calculate the difference in latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Calculate the distance using the Haversine formula
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Distance in kilometers
    distance = R * c

    return distance


def interquartile(df, lower_quantile=0.1, upper_quantile=0.9):
    """
    Filter a DataFrame based on interquartile ranges for each column.

    Parameters:
        df (pandas DataFrame): DataFrame to be filtered.
        lower_quantile (float): Lower quantile threshold.
        upper_quantile (float): Upper quantile threshold.

    Returns:
        pandas DataFrame: Filtered DataFrame.
    """
    filtered_df = df[
        (df >= df.quantile(lower_quantile)) &
        (df <= df.quantile(upper_quantile))
    ]

    return filtered_df.dropna(axis=1)
