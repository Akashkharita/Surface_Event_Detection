import os
import sys
import time
import json
import yaml
import h5py
import tsfel
import random
import calendar
import obspy
import seaborn as sns
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import stats
from scipy import signal
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.mass_downloader import CircularDomain, Restrictions, MassDownloader
from obspy.signal.filter import envelope
from matplotlib import lines as mlines
from pytz import timezone
from dateutil import parser
from zenodo_get import zenodo_get
from tsfel import time_series_features_extractor

import sys
sys.path.append('../src')
import seis_feature

from utils import apply_cosine_taper, butterworth_filter, resample_array

import warnings
# Configure display settings and ignore warnings
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

# Set up default values
client = Client('IRIS')
cfg_file = tsfel.get_features_by_domain()
location = '*'
original_sr = 100
samp_freq = 100
num_corners = 4

# Global variables
starttime = []
stations_id = []
dur = []
stride = []
new_sr = []
scaler_params = []
best_model = []
lowcut = []
highcut = []
win = []
filename = []


client = Client('IRIS')
def event_detection(starttime=starttime, stations_id=stations_id, dur=dur, stride=stride, new_sr=new_sr,
                    original_sr=original_sr, low =lowcut, high =highcut, num_corners=num_corners,
                    location=location, samp_freq=samp_freq, win=win, filename = filename, top_20 = False):
    
    
    
    
    scaler_params = pd.read_csv('scaler_params_phy_man_'+filename+'.csv')
    best_model = load('best_rf_model_all_features_phy_man_'+filename+'.joblib')
    
    if top_20:
        scaler_params = pd.read_csv('scaler_params_phy_man_top_20_'+filename+'.csv')
        best_model = load('best_rf_model_all_features_phy_man_top_20_'+filename+'.joblib')
        
        

    st_data_full, st_overall, st_overall_data, st_overall_times = [], [], [], []
    result_stns, index_stns, prob_stns = [], [], []

    for stn_id in tqdm(stations_id):
        stn, network = stn_id.split('.')[1], stn_id.split('.')[0]
        st = []

        # Try to fetch waveform data from different channels
        for channel in ['EHZ', 'BHZ', 'HHZ']:
            try:
                st += client.get_waveforms(starttime=starttime, endtime=starttime + dur, station=stn,
                                           network=network, channel=channel, location=location)
                break
            except:
                continue

        
        
        
        try:

            # Detrend and resample the data
            st = obspy.Stream(st).detrend()
            print(st)
            st.resample(samp_freq)

            times = st[0].times()
            st_data_full = np.hstack([tr.data for tr in st])

            # Handle multiple streams
            for i in range(1, len(st)):
                diff = st[i].stats.starttime - st[i-1].stats.endtime
                times = np.hstack((times, st[i].times() + times[-1] + diff))

            # Store data and times
            st_overall_data.append(st_data_full)
            st_overall.append(st)
            st_overall_times.append(times)

            trace_data = [st_data_full[i:i+int(win*samp_freq)] for i in tqdm(range(0, len(st_data_full), stride*samp_freq)) if len(st_data_full[i:i+int(win*samp_freq)]) == int(win*samp_freq)]
            trace_times = [times[i] for i in tqdm(range(0, len(st_data_full), stride*samp_freq)) if len(st_data_full[i:i+int(win*samp_freq)]) == int(win*samp_freq)]

            # Process trace data
            trace_data = np.array(trace_data)
            tapered = apply_cosine_taper(trace_data, 10)      
            filtered_data = np.array(butterworth_filter(tapered, low, high, original_sr, num_corners, 'bandpass'))
            normalized_data = filtered_data / np.max(abs(filtered_data), axis=1)[:, np.newaxis]
            resampled_data = np.array([resample_array(arr, original_sr, new_sr) for arr in normalized_data])


            result, prob, index = [], [], []

            for i in range(len(resampled_data)):
                try:

                    #tsfel_features = time_series_features_extractor(cfg_file, norm[i], fs=new_sr, verbose=0)
                    columns = scaler_params['Feature'].values

                    scaler_params.index = columns

                    physical_features = seis_feature.FeatureCalculator(resampled_data[i], fs=new_sr, selected_features = columns).compute_features()
                    final_features =  physical_features            #pd.concat([tsfel_features, physical_features], axis=1)
                    #columns = scaler_params['Feature'].values
                    features = final_features.loc[:, columns]

                    # Scale features
                    for j in range(len(columns)):
                        features[columns[j]] = (features[columns[j]] - scaler_params.loc[columns[j],'Mean'])/scaler_params.loc[columns[j], 'Std Dev']

                    # Add time features
                    if len(columns) < 30:
                        features['hod'] = (starttime).hour - 8
                    else:
                        features['hod'] = (starttime).hour - 8
                        features['dow'] = (starttime).weekday
                        features['moy'] = (starttime).month

                    # Predict results
                    result.append(best_model.predict(features))
                    prob.append(best_model.predict_proba(features))
                    index.append(i)

                except:
                    pass



            result_stns.append(result)
            index_stns.append(index)
            prob_stns.append(prob)
            
        except:
            pass


    return result_stns, index_stns, prob_stns, st_overall, st_overall_data, st_overall_times





st_overall_data = []
st_overall_times = []
st_overall = []
result_stns = []
index_stns = []
prob_stns = []
before = []
dur = []


def plot_detection_results(st_overall_data=st_overall_data, st_overall_times=st_overall_times, st_overall=st_overall,
                           result_stns=result_stns, index_stns=index_stns, prob_stns=prob_stns, xlim=[0, dur],
                           ev_markers=[before, dur], shift=stride, win=win, filename=filename):

    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 20
    fig, axs = plt.subplots(len(index_stns)+1, 1, figsize=(15, 3*(len(index_stns)+1)))

    colors = ['black', 'blue', 'white', 'red']
    legend_elements = [
        mlines.Line2D([], [], marker='o', color='red', mec='k', label='Prob (Su)', markersize=10),
        mlines.Line2D([], [], marker='o', color='k', mec='k', label='Prob (Eq)', markersize=10),
        mlines.Line2D([], [], marker='o', color='white', mec='k', label='Prob (No)', markersize=10),
        mlines.Line2D([], [], marker='o', color='blue', mec='k', label='Prob (Exp)', markersize=10)
    ]

    for k in range(len(index_stns)):
        axs[k].plot(st_overall_times[k], st_overall_data[k] / np.max(abs(st_overall_data[k])))
        axs[k].set_title(st_overall[k][0].id, fontsize=20)

        for i in range(len(index_stns[k])):
            axs[k].axvline(shift * index_stns[k][i] + (win/2), ls='--', color=colors[int(result_stns[k][i])], alpha=0.6)

        for i in range(len(index_stns[k])):
            color = colors[int(result_stns[k][i])]
            axs[k].scatter(shift * np.array(index_stns[k])[i] + (win/2), np.array(prob_stns[k])[:, :, int(result_stns[k][i])][i], ec='k', marker='o', c=color, s=100, zorder=5)

        axs[k].legend(handles=legend_elements, loc='upper right', fontsize=12)
        axs[k].set_xlabel(f'Time(s) since {str(starttime).split(".")[0]}', fontsize=20)
        axs[k].set_xlim(xlim[0], xlim[1])
        axs[k].axvline(ev_markers[0], ls='-', c='k', lw=2)
        axs[k].axvline(ev_markers[1], ls='-', c='k', lw=2)
    
    max_size = max(max(arr) for arr in index_stns)+1

    eq_probs = np.zeros([len(index_stns), max_size])
    exp_probs = np.zeros([len(index_stns), max_size])
    su_probs = np.zeros([len(index_stns), max_size])
    no_probs = np.zeros([len(index_stns), max_size])

    for i in range(len(index_stns)):
        for j in range(len(index_stns[i])):
            try:
                eq_probs[i, index_stns[i][j]] = prob_stns[i][j][0][0]
                exp_probs[i, index_stns[i][j]] = prob_stns[i][j][0][1]
                no_probs[i, index_stns[i][j]] = prob_stns[i][j][0][2]
                su_probs[i, index_stns[i][j]] = prob_stns[i][j][0][3]
            except IndexError as e:
                print(f"IndexError: {e}")

    mean_eq_probs = np.mean(eq_probs, axis=0)
    mean_exp_probs = np.mean(exp_probs, axis=0)
    mean_no_probs = np.mean(no_probs, axis=0)
    mean_su_probs = np.mean(su_probs, axis=0)

    axs[len(index_stns)].plot(shift * np.arange(max_size) + (win/2), mean_eq_probs, 'k', label='Prob (Eq)', lw=3, marker = 'o')
    axs[len(index_stns)].plot(shift * np.arange(max_size) + (win/2), mean_exp_probs, 'b', label='Prob (Exp)', lw=3, marker = 'o')
    axs[len(index_stns)].plot(shift * np.arange(max_size) + (win/2), mean_no_probs, 'w', label='Prob (No)', lw=3, marker = 'o')
    axs[len(index_stns)].plot(shift * np.arange(max_size) + (win/2), mean_su_probs, 'r', label='Prob (Su)', lw=3, marker = 'o')
    axs[len(index_stns)].set_xlabel(f'Time(s) since {str(starttime).split(".")[0]}', fontsize=20)
    axs[len(index_stns)].axvline(ev_markers[0], ls='-', c='k', lw=2)
    axs[len(index_stns)].axvline(ev_markers[1], ls='-', c='k', lw=2)
    axs[len(index_stns)].legend(loc='upper right', fontsize=12)
    axs[len(index_stns)].set_xlim(xlim[0], xlim[1])

    fig.tight_layout()
    plt.savefig(f"{filename}.jpg", dpi=400)
    plt.show()

    
    
def plot_detection_results(st_overall_data = st_overall_data, st_overall_times = st_overall_times, st_overall = st_overall, result_stns = result_stns, index_stns = index_stns, prob_stns = prob_stns, xlim = [0,dur], ev_markers = [before,dur], shift = stride, win = win, filename = filename):
    

    
    plt.rcParams['xtick.labelsize'] = 16  # Font size for xtick labels
    plt.rcParams['ytick.labelsize'] = 20  # Font size for ytick labels

    fig, axs = plt.subplots(len(index_stns)+1, 1, figsize=(15, 3*(len(index_stns)+1)))

    for k in range(len(index_stns)):

        ## This is plotting the normalized data
        axs[k].plot(st_overall_times[k], st_overall_data[k] / np.max(abs(st_overall_data[k])))

        ## Setting the title of the plot
        axs[k].set_title(st_overall[k][0].id, fontsize=20)

        ## These are the colors of detection window. 
        colors = ['black', 'blue', 'white', 'red']
        for i in range(len(index_stns[k])):
            axs[k].axvline(shift * index_stns[k][i] + (win/2), ls='--', color=colors[int(result_stns[k][i])], alpha = 0.6)
            
        # Plot circles on top of the line plot
        for i in range(len(index_stns[k])):
            if result_stns[k][i] == 3:
                axs[k].scatter(shift * np.array(index_stns[k])[i] + (win/2), np.array(prob_stns[k])[:, :, 3][i], ec='k', marker='o', c='red', s=100, zorder=5)
            elif result_stns[k][i] == 0:
                axs[k].scatter(shift * np.array(index_stns[k])[i] + (win/2), np.array(prob_stns[k])[:, :, 0][i], ec='k', marker='o', c='black', s=100, zorder=5)
            elif result_stns[k][i] == 1:
                axs[k].scatter(shift * np.array(index_stns[k])[i] + (win/2), np.array(prob_stns[k])[:, :, 1][i], ec='k', marker='o', c='blue', s=100, zorder=5)
            else:
                axs[k].scatter(shift * np.array(index_stns[k])[i] + (win/2), np.array(prob_stns[k])[:, :, 2][i], ec='k', marker='o', c='white', s=100, zorder=5)

        # Create custom legend for circular markers
        legend_elements = [
           mlines.Line2D([], [], marker='o', color='red', mec = 'k', label='Prob (Su)', markersize=10),
            mlines.Line2D([], [], marker='o', color='k', mec = 'k', label='Prob (Eq)', markersize=10), 
            mlines.Line2D([], [], marker='o', color='white', mec = 'k',  label='Prob (No)', markersize=10),
             mlines.Line2D([], [], marker='o', color='blue', mec = 'k',  label='Prob (Exp)', markersize=10)
        ]
        axs[k].legend(handles=legend_elements, loc='upper right', fontsize=12)

        axs[k].set_xlabel('Time(s) since ' + str(starttime).split('.')[0], fontsize=20)
        axs[k].set_xlim(xlim[0], xlim[1])  # Set x-axis limits if needed
        axs[k].axvline(ev_markers[0], ls = '-', c = 'k', lw = 2)
        axs[k].axvline(ev_markers[1], ls = '-', c = 'k', lw = 2)
        
    
    
    
    # Finding the size of the biggest array
    max_size = max(max(arr) for arr in index_stns)+1

    print(max_size)

    # initializing different lists. 
    eq_probs = np.zeros([len(index_stns), max_size])
    exp_probs = np.zeros([len(index_stns), max_size])
    su_probs = np.zeros([len(index_stns), max_size])
    no_probs = np.zeros([len(index_stns), max_size])



    # saving the probabilities. 

    for i in range(len(index_stns)):
        for j in range(len(index_stns[i])):
            
            try:
                #print(f"Processing i={i}, j={j}")
                #print(f"index_stns[i][j] = {index_stns[i][j]}")
                #print(f"prob_stns[i][j] = {prob_stns[i][j]}")

                eq_probs[i, index_stns[i][j]] = prob_stns[i][j][0][0]
                exp_probs[i, index_stns[i][j]] = prob_stns[i][j][0][1]
                no_probs[i, index_stns[i][j]] = prob_stns[i][j][0][2]
                su_probs[i, index_stns[i][j]] = prob_stns[i][j][0][3]

            except IndexError as e:
                            print(f"IndexError: {e}")
                            #print(f"eq_probs shape: {eq_probs.shape}")
                            #print(f"index_stns[i]: {index_stns[i]}")
                            #print(f"prob_stns shape: {len(prob_stns)}, {len(prob_stns[i])}, {len(prob_stns[i][j][0])}")         
            
            
    mean_eq_probs = np.mean(eq_probs, axis = 0)
    mean_exp_probs = np.mean(exp_probs, axis = 0)
    mean_no_probs = np.mean(no_probs, axis = 0)
    mean_su_probs = np.mean(su_probs, axis = 0)
    
    
    axs[-1].set_title('Mean probabilities of each event type across all the stations', fontsize = 20)
    axs[-1].scatter(shift*np.arange(max_size)+(win/2), mean_eq_probs, marker = 'o', c = 'k', s = 100, ec = 'k')
    axs[-1].scatter(shift*np.arange(max_size)+(win/2), mean_exp_probs, marker = 'o', c = 'b', s = 100, ec = 'k')
    axs[-1].scatter(shift*np.arange(max_size)+(win/2),mean_no_probs, marker = 'o', c = 'white', s = 100, ec = 'k')
    axs[-1].scatter(shift*np.arange(max_size)+(win/2),mean_su_probs, marker = 'o', c = 'r', s = 100, ec= 'k')
    axs[-1].set_xlim(xlim[0], xlim[1])
    axs[-1].set_ylim(0, 1)
    axs[-1].set_ylabel('Mean Probability', fontsize = 20)
    axs[-1].legend(handles=legend_elements, loc='upper right', fontsize=12)
    fig.suptitle('Results for the model: '+filename, fontsize = 25)
    
    
    plt.tight_layout()  # Adjust subplots to avoid overlap
    plt.show()
