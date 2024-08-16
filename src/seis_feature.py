import numpy as np
import pandas as pd
from glob import glob 
from tqdm import tqdm
import seaborn as sns 

# for converting the text file containing the quarry locations into csv file
import csv

# for computing the geographical distance between two points 
import math


from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from datetime import datetime
import h5py
from sklearn.preprocessing import LabelEncoder
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
import obspy
from obspy.geodetics.base import gps2dist_azimuth, gps2dist_azimuth
from obspy.clients.fdsn import Client
import time
pd.set_option('display.max_columns', None)
from joblib import dump, load
from obspy.signal.filter import envelope
import tsfel



#from seis_feature import compute_physical_features
from tsfel import time_series_features_extractor, get_features_by_domain
from datetime import timedelta
import os
import sys


# Get the absolute path of the directory two levels up
#two_levels_up = os.path.abspath(os.path.join(os.getcwd(), "../.."))

# Append the 'src' directory located two levels up to the system path
#sys.path.append(os.path.join(two_levels_up, 'src'))
#from utils import apply_cosine_taper
#from utils import butterworth_filter
#from utils import resample_array



import pickle
from zenodo_get import zenodo_get

from multiprocessing import Pool, cpu_count
from scipy.signal import resample

from scipy import signal
import scipy
import numpy as np
import obspy
import pandas as pd
import tsfel
from scipy import signal
from scipy.fft import fft, fftfreq
import numpy as np
from scipy.signal import hilbert
from sklearn import metrics



def RSAM(data, samp_rate, datas, freq, Nm, N):
    filtered_data = obspy.signal.filter.bandpass(data, freq[0], freq[1], samp_rate)
    filtered_data = abs(filtered_data[:Nm])
    datas.append(filtered_data.reshape(-1,N).mean(axis=-1)*1.e9) # we should remove the append
    return(datas)

def DSAR(data, samp_rate, datas, freqs_names, freqs, Nm, N):
    # compute dsar
    data = scipy.integrate.cumtrapz(data, dx=1./100, initial=0) # vel to disp
    data -= np.mean(data) # detrend('mean')
    j = freqs_names.index('mf')
    mfd = obspy.signal.filter.bandpass(data, freqs[j][0], freqs[j][1], samp_rate)
    mfd = abs(mfd[:Nm])
    mfd = mfd.reshape(-1,N).mean(axis=-1)
    j = freqs_names.index('hf')
    hfd = obspy.signal.filter.bandpass(data, freqs[j][0], freqs[j][1], samp_rate)
    hfd = abs(hfd[:Nm])
    hfd = hfd.reshape(-1,N).mean(axis=-1)
    dsar = mfd/hfd
    datas.append(dsar)
    return(datas, dsar)

def nDSAR(dsar):
    return dsar/scipy.stats.zscore(dsar)




class FeatureCalculator:
    def __init__(self, data, fs=100, envfilter=True, freq_bands=[[0.1, 1], [1, 3], [3, 10], [10, 20], [20, 50]], env_filt=[0.5], selected_features=None):
        self.data = data
        self.fs = fs
        self.envfilter = envfilter
        self.freq_bands = freq_bands
        self.env_filt = env_filt
        self.selected_features = selected_features
        self.t = np.linspace(0, int(len(data)/fs), len(data))

        self.ft = abs(fft(data))
        self.freq = fftfreq(len(data), d=1/fs)
        self.ft = self.ft[:len(self.ft)//2]
        self.freq = self.freq[:len(self.freq)//2]
        self.norm_ft = self.ft / np.sum(self.ft)

        self.f, self.t1, self.Sxx = signal.spectrogram(data, fs=fs)
        self.split_indices = [0, len(self.t1) // 4, 2 * (len(self.t1) // 4), 3 * (len(self.t1) // 4), 4 * (len(self.t1) // 4)]
        self.Sq = [abs(self.Sxx[:, self.split_indices[i]:self.split_indices[i+1]]) for i in range(len(self.split_indices) - 1)]

        self.auto = np.correlate(data, data, 'same')
        self.env = self.compute_envelope(data)
        if envfilter:
            sos = signal.butter(4, env_filt, 'lp', fs=fs, output='sos')
            self.env = signal.sosfilt(sos, self.env)
        self.l = np.nanmax(self.env) - ((np.nanmax(self.env) / (self.t[-1] - self.t[np.nanargmax(self.env)])) * (self.t))

        self.feature_functions = {
            'Window_Length': lambda: self.t[-1] - self.t[0],
            'RappMaxMean': lambda: np.nanmax(self.env) / np.nanmean(self.env),
            'RappMaxMedian': lambda: np.nanmax(self.env) / np.nanmedian(self.env),
            'AsDec': lambda: (self.t[np.argmax(self.env)] - self.t[0]) / (self.t[-1] - self.t[np.argmax(self.env)]),
            'KurtoSig': lambda: scipy.stats.kurtosis(self.data),
            'KurtoEnv': lambda: scipy.stats.kurtosis(self.env),
            'SkewSig': lambda: scipy.stats.skew(self.data),
            'SkewEnv': lambda: scipy.stats.skew(self.env),
            'CorPeakNumber': lambda: len(scipy.signal.find_peaks(self.auto)[0]),
            'Energy1/3Cor': lambda: np.trapz(y=self.auto[0:int(len(self.auto)/3)]),
            'Energy2/3Cor': lambda: np.trapz(y=self.auto[int(len(self.auto)/3):len(self.auto)]),
            'int_ratio': lambda: np.trapz(y=self.auto[0:int(len(self.auto)/3)]) / np.trapz(y=self.auto[int(len(self.auto)/3):len(self.auto)]),
            'RMSDecPhaseLine': lambda: np.sqrt(np.nanmean((self.env - self.l)**(2))),
            'MeanFFT': lambda: np.nanmean(self.ft),
            'MaxFFT': lambda: np.nanmax(self.ft),
            'FMaxFFT': lambda: self.freq[np.nanargmax(self.ft)],
            'MedianFFT': lambda: np.nanmedian(self.norm_ft),
            'VarFFT': lambda: np.nanvar(self.norm_ft),
            'FCentroid': lambda: np.dot(self.freq, self.ft) / (np.sum(self.ft)),
            'Fquart1': lambda: np.dot(self.freq[0:len(self.ft)//4], self.ft[0:len(self.ft)//4]) / (np.sum(self.ft[0:len(self.ft)//4])),
            'Fquart3': lambda: np.dot(self.freq[len(self.ft)//2:3*len(self.ft)//4], self.ft[len(self.ft)//2:3*len(self.ft)//4]) / (np.sum(self.ft[len(self.ft)//2:3*len(self.ft)//4])),
            'NPeakFFT': lambda: len(signal.find_peaks(self.ft, height=0.75*np.nanmax(self.ft))[0]),
            'MeanPeaksFFT': lambda: np.nanmean(self.ft[signal.find_peaks(self.ft, height=0)[0]]),
            'E1FFT': lambda: np.trapz(y=self.ft[:len(self.ft)//4], x=self.freq[:len(self.ft)//4]),
            'E2FFT': lambda: np.trapz(y=self.ft[len(self.ft)//4:len(self.ft)//2], x=self.freq[len(self.ft)//4:len(self.ft)//2]),
            'E3FFT': lambda: np.trapz(y=self.ft[len(self.ft)//2:int(3*len(self.ft)//4)], x=self.freq[len(self.ft)//2:int(3*len(self.ft)//4)]),
            'E4FFT': lambda: np.trapz(y=self.ft[int(3*len(self.ft)//4):len(self.ft)], x=self.freq[int(3*len(self.ft)//4):len(self.ft)]),
            'Gamma1': lambda: np.dot(self.freq, self.ft**(2)) / np.sum(self.ft**(2)),
            'Gamma2': lambda: (np.dot(self.freq**(2), self.ft**(2)) / np.sum(self.ft**(2)))**(0.5),
            'Gamma': lambda: ((np.dot(self.freq, self.ft**(2)) / np.sum(self.ft**(2)))**(2) - ((np.dot(self.freq**(2), self.ft**(2)) / np.sum(self.ft**(2)))**(0.5))**(2))**(0.5),
            'KurtoMaxDFT': lambda: scipy.stats.kurtosis(np.nanmax(abs(self.Sxx), axis=0)),
            'KurtoMedianDFT': lambda: scipy.stats.kurtosis(np.nanmedian(abs(self.Sxx), axis=0)),
            'MaxOverMeanDFT': lambda: np.nanmean(np.nanmax(abs(self.Sxx), axis=0) / np.nanmean(abs(self.Sxx), axis=0)),
            'MaxOverMedianDFT': lambda: np.nanmean(np.nanmax(abs(self.Sxx), axis=0) / np.nanmedian(abs(self.Sxx), axis=0)),
            'NbrPeaksMaxDFT': lambda: len(signal.find_peaks(np.nanmax(abs(self.Sxx), axis=0))[0]),
            'NbrPeaksMeanDFT': lambda: len(signal.find_peaks(np.nanmean(abs(self.Sxx), axis=0))[0]),
            'NbrPeaksMedianDFT': lambda: len(signal.find_peaks(np.nanmedian(abs(self.Sxx), axis=0))[0]),
            '45/46': lambda: len(signal.find_peaks(np.nanmax(abs(self.Sxx), axis=0))[0]) / len(signal.find_peaks(np.nanmean(abs(self.Sxx), axis=0))[0]),
            '45/47': lambda: len(signal.find_peaks(np.nanmax(abs(self.Sxx), axis=0))[0]) / len(signal.find_peaks(np.nanmedian(abs(self.Sxx), axis=0))[0]),
            'NbrPeaksCentralFreq': lambda: len(signal.find_peaks(np.dot(self.f, abs(self.Sxx)) / np.sum(abs(self.Sxx), axis=0))[0]),
            'NbrPeaksMaxFreq': lambda: len(signal.find_peaks(self.f[np.nanargmax(abs(self.Sxx), axis=0)])[0]),
            '50/51': lambda: len(signal.find_peaks(np.dot(self.f, abs(self.Sxx)) / np.sum(abs(self.Sxx), axis=0))[0]) / len(signal.find_peaks(self.f[np.nanargmax(abs(self.Sxx), axis=0)])[0]),
            'DistMaxMeanFreqDTF': lambda: np.nanmean(np.nanmax(abs(self.Sxx), axis=0) - np.nanmean(abs(self.Sxx), axis=0)),
            'DistMaxMedianFreqDTF': lambda: np.nanmean(np.nanmax(abs(self.Sxx), axis=0) - np.nanmedian(abs(self.Sxx), axis=0)),
            'DistQ2Q1DFT': lambda: np.nanmean(np.dot(self.f, self.Sq[1]) / np.sum(self.Sq[1], axis=0) - np.dot(self.f, self.Sq[0]) / np.sum(self.Sq[0], axis=0)),
            'DistQ3Q2DFT': lambda: np.nanmean(np.dot(self.f, self.Sq[2]) / np.sum(self.Sq[2], axis=0) - np.dot(self.f, self.Sq[1]) / np.sum(self.Sq[1], axis=0)),
            'DistQ3Q1DFT': lambda: np.nanmean(np.dot(self.f, self.Sq[2]) / np.sum(self.Sq[2], axis=0) - np.dot(self.f, self.Sq[0]) / np.sum(self.Sq[0], axis=0)),
            'Peak_Envelope_Amplitude': lambda: np.nanmax(self.env),
            'Average_Envelope_Amplitude': lambda: np.nanmean(self.env),
            'Envelope_Area': lambda: metrics.auc(self.t, self.env),
            'Envelope_Velocity': lambda: (metrics.auc(self.t, self.env)) / (self.t[-1] - self.t[0]),
            'Envelope_Rise_Time': lambda: self.t[np.argmax(self.env)] - self.t[0]
        }

        # Add lambda functions for frequency bands
        for i, (low, high) in enumerate(self.freq_bands):
            try:
                sos = signal.butter(N=4, Wn=[low, high], btype='bp', fs=self.fs, output='sos')
                filtered = signal.sosfilt(sos, self.data)
                env = self.compute_envelope(filtered)
                self.feature_functions['E_' + str(low) + '_' + str(high)] = lambda env=env, t=self.t: np.log10(np.trapz(y=abs(env), x=t))
                self.feature_functions['Kurto_' + str(low) + '_' + str(high)] = lambda filtered=filtered: scipy.stats.kurtosis(filtered)
            except Exception as e:
                
                self.feature_functions['E_' + str(low) + '_' + str(high)] = lambda: 0
                self.feature_functions['Kurto_' + str(low) + '_' + str(high)] = lambda: 0
                #print(f"Error in freq band {low}-{high}: {e}")

    def compute_envelope(self, data):
        analytic_signal = signal.hilbert(data)
        amplitude_envelope = np.abs(analytic_signal)
        return amplitude_envelope

    def compute_features(self):
        if self.selected_features is None:
            self.selected_features = list(self.feature_functions.keys())
        feature_values = {}
        for feature_name in self.selected_features:
            if feature_name in self.feature_functions:
                feature_values[feature_name] = self.feature_functions[feature_name]()
        feature_df = pd.DataFrame(data=[feature_values], columns=self.selected_features)
        return feature_df

