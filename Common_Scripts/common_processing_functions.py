import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import obspy
from obspy.signal.filter import envelope
from obspy.clients.fdsn import Client
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc, classification_report, confusion_matrix, f1_score
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#from xgboost import XGBClassifier
from sklearn.decomposition import PCA
#from imblearn.under_sampling import RandomUnderSampler
from scipy import stats, signal
from sklearn.datasets import load_iris
from sklearn.metrics import precision_score, recall_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
pd.set_option('display.max_columns', None)
from obspy.geodetics.base import gps2dist_azimuth
from datetime import datetime, timedelta
import time
#import lightgbm as lgb
import re
import tsfel
import random
import calendar
import concurrent.futures


# We need to change the path to import the seis feature library. 

import sys
sys.path.append('../Feature_Extraction_Scripts/Physical_Feature_Extraction_Scripts/')
import seis_feature
from seis_feature import compute_physical_features
from multiprocessing import Pool
from tsfel import time_series_features_extractor, get_features_by_domain
from functools import partial
import multiprocessing




def apply_cosine_taper(arrays, taper_percent=10):
    tapered_arrays = []
    
    #print(arrays.shape)
    num_samples = arrays.shape[1]  # Assuming each sub-array has the same length
    
    for array in arrays:
        

        taper_length = int(num_samples * taper_percent / 100)
        taper_window = np.hanning(2 * taper_length)
        
     
        tapered_array = array.copy()
        tapered_array[:taper_length] = tapered_array[:taper_length] * taper_window[:taper_length]
        tapered_array[-taper_length:] = tapered_array[-taper_length:] * taper_window[taper_length:]
        
        tapered_arrays.append(tapered_array)
    
    return np.array(tapered_arrays)              
              
    
    

def butterworth_filter(arrays, lowcut = 1, highcut = 10, fs = 100, num_corners = 4, filter_type='bandpass'):
    """
    Apply a Butterworth filter (bandpass, highpass, or lowpass) to each array in an array of arrays.

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
        # Normalize the frequency values to Nyquist frequency (0.5*fs)
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

        # Apply the filter to the data using lfilter
        filtered_data = signal.lfilter(b, a, data)

        filtered_arrays.append(filtered_data)

    return filtered_arrays

              
    
    
    
def plot_detection_results(st_overall_data = st_overall_data, st_overall_times = st_overall_times, st_overall = st_overall, result_stns = result_stns, index_stns = index_stns, prob_stns = prob_stns, xlim = [0,300]):
    plt.rcParams['xtick.labelsize'] = 16  # Font size for xtick labels
    plt.rcParams['ytick.labelsize'] = 20  # Font size for ytick labels

    fig, axs = plt.subplots(len(st_overall_data), 1, figsize=(15, 3*len(st_overall_data)))

    for k in range(len(st_overall_data)):

        ## This is plotting the normalized data
        axs[k].plot(st_overall_times[k], st_overall_data[k] / np.max(abs(st_overall_data[k])))

        ## Setting the title of the plot
        axs[k].set_title(st_overall[k][0].id, fontsize=20)

        ## These are the colors of detection window. 
        colors = ['black', 'blue', 'white', 'red']
        for i in range(len(index_stns[k])):
            axs[k].axvline(30 * index_stns[k][i] + 75, ls='--', color=colors[int(result_stns[k][i])], alpha = 0.6)
        for i in range(len(index_stns[k])):
            if result_stns[k][i] == 3:
                axs[k].scatter(30 * np.array(index_stns[k])[i] + 75, np.array(prob_stns[k])[:, :, 3][i], ec='k', marker='o', c='red', s=100)
            elif result_stns[k][i] == 0:
                axs[k].scatter(30 * np.array(index_stns[k])[i] + 75, np.array(prob_stns[k])[:, :, 3][i], ec='k', marker='o', c='black', s=100)
            elif result_stns[k][i] == 1:
                axs[k].scatter(30 * np.array(index_stns[k])[i] + 75, np.array(prob_stns[k])[:, :, 3][i], ec='k', marker='o', c='blue', s=100)
            else:
                axs[k].scatter(30 * np.array(index_stns[k])[i] + 75, np.array(prob_stns[k])[:, :, 3][i], ec='k', marker='o', c='white', s=100,)

        # Create custom legend for circular markers
        legend_elements = [
            mlines.Line2D([], [], marker='o', color='red', label='Prob (Su)', markersize=10)
        ]
        axs[k].legend(handles=legend_elements, loc='upper right', fontsize=12)

        axs[k].set_xlabel('Time(s) since ' + str(starttime).split('.')[0], fontsize=20)
        axs[k].set_xlim(xlim[0], xlim[1])  # Set x-axis limits if needed

    plt.tight_layout()  # Adjust subplots to avoid overlap
    plt.show()

    