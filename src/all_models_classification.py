import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import obspy
# from tqdm import tqdm
from glob import glob
# import time
import random
import os
import sys
from datetime import datetime
from tqdm import tqdm

from scipy import stats,signal


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


import numpy as np
import scipy.signal as signal

from matplotlib import lines as mlines

import sys
sys.path.append('../src')
from utils import apply_cosine_taper, butterworth_filter, resample_array


import numpy as np
import torch
import torch.nn.functional as F
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm


from joblib import dump, load


import os
# Specify the directory containing the module
module_path = os.path.abspath(os.path.join('..', 'src'))

# Add the directory to sys.path
if module_path not in sys.path:
    sys.path.append(module_path)

import seis_feature
from utils import apply_cosine_taper, butterworth_filter, resample_array
 

from deep_learning_architectures import MyCNN_1d
from deep_learning_architectures import MyCNN_2d
from deep_learning_architectures import SeismicCNN_2d
from deep_learning_architectures import MyResCNN2D


## setting up some important parameters (not to be changed)
num_channels = 3
dropout = 0.9
# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


















# Initialize FDSN client (e.g., IRIS)
client = Client("IRIS")

stations_id = ['CC.WOW', 'CC.TAVI',  'CC.GNOB', 'CC.ARAT', 'CC.TABR', 'UW.RER']


## taking all the available location
location = "*"

# starttime 
start_time = obspy.UTCDateTime(2024, 8, 15, 17, 39, 52) - 0

# endtime
end_time = start_time + 300

# length
signal_length = end_time - start_time

# channel_patterns
channel_patterns = ["EH", "BH", "HH"]

os = 100 # original sampling rate
fs = 50 # new sampling rate
stride = 10*(os)
window_length = 100
model = []
one_d = False


lowpass = 1
highpass = 20
orig_sr = 100
new_sr = 50




def get_station_inventory(network, station, location, start_time, end_time, client):
    """
    Fetch station inventory data from FDSN client.
    """
    try:
        inventory = client.get_stations(network=network, station=station, location=location,
                                        channel="*H*", starttime=start_time, endtime=end_time, level="channel")
        return inventory
    except Exception as e:
        print(f"Error fetching station data for {station}: {e}")
        return None


def has_three_components(station, prefix):
    """
    Check if a station has all three components (E, N, Z) for a given channel prefix.
    """
    channels = [chan.code for chan in station.channels if chan.code.startswith(prefix)]
    return set([f"{prefix}E", f"{prefix}N", f"{prefix}Z"]).issubset(set(channels))


def fetch_waveform_data(client, network, station, location, prefix, start_time, end_time):
    """
    Fetch waveform data from the client for a station with the given channel prefix.
    """
    try:
        st = client.get_waveforms(network=network, station=station, location=location,
                                  channel=f"{prefix}?", starttime=start_time, endtime=end_time)
        return st
    except Exception as e:
        print(f"Error fetching waveform data for {station}: {e}")
        return Stream()


def process_station(network, station, location, start_time, end_time, channel_patterns, client, plot_data=False):
    """
    Process the station to fetch waveform data and return the combined stream.
    """
    inventory = get_station_inventory(network, station, location, start_time, end_time, client)
    if inventory is None:
        return Stream()

    st_big = []
    for net in inventory:
        for sta in net:
            for prefix in channel_patterns:
                # checks if it has all three components for a given channel prefix, if not it will move to other.
                if has_three_components(sta, prefix):
                    #print(f"Station {sta.code} has all 3 components for {prefix} channels.")
                    st = fetch_waveform_data(client, net.code, sta.code, location, prefix, start_time, end_time)
                    st_big += st
                    if plot_data:
                        st.plot()
                    break  # Exit loop after finding one valid channel pattern

    return Stream(st_big)


def taper_and_bandpass_filter(data, lowcut= lowpass, highcut= highpass, fs= orig_sr):
    """
    Apply a bandpass filter to the input data.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    
    # Apply taper and filter
    taper = signal.windows.tukey(data.shape[-1], alpha=0.1)
    data = np.array([np.multiply(taper, row) for row in data])
    data = np.array([signal.filtfilt(b, a, row) for row in data])
    
    return data


def reshape_and_resample(st, signal_length, fs=100):
    """
    Detrend, resample, and reshape the data for each station.
    """
    st.detrend(type = 'linear')
    st = np.array([signal.resample(row, int(signal_length*fs)) for row in st])
    
    reshaped_data = np.array(st).reshape(len(st) // 3, 3, -1)[0].reshape(1,3,-1)
    return reshaped_data



def extract_spectrograms(waveforms = [], fs = new_sr, nperseg=256, overlap=0.5):
    noverlap = int(nperseg * overlap)  # Calculate overlap

    # Example of how to get the shape of one spectrogram
    f, t, Sxx = signal.spectrogram(waveforms[0, 0], nperseg=nperseg, noverlap=noverlap, fs=fs)

    # Initialize an array of zeros with the shape: (number of waveforms, channels, frequencies, time_segments)
    spectrograms = np.zeros((waveforms.shape[0], waveforms.shape[1], len(f), len(t)))

    for i in range(waveforms.shape[0]):  # For each waveform
        for j in range(waveforms.shape[1]):  # For each channel
            _, _, Sxx = signal.spectrogram(waveforms[i, j], nperseg=nperseg, noverlap=noverlap, fs=fs)
            spectrograms[i, j] = Sxx  # Fill the pre-initialized array

    #print(spectrograms.shape)
    return spectrograms




def compute_window_probs(stations_id = stations_id, location = location, start_time = start_time, 
                        end_time = end_time, channel_patterns = channel_patterns, client = client,
                        orig_sr = orig_sr, new_sr = new_sr, window_length = window_length, lowpass = lowpass, stride = stride, 
                        highpass = highpass, one_d = one_d, model = model, model_type = 'dl', filename = 'P_10_30_F_05_15_50'):

    big_station_wise_probs = []
    big_reshaped_data = []
    big_station_ids = []
    signal_length = end_time - start_time
    
    # Main processing loop for stations
    for stn_id in tqdm(stations_id):

        ## extract the network and the stations UW.STAR --> UW, STAR
        network, station = stn_id.split('.')




        # preallocating array saves computation time wherever possible. 
        station_ids = []


        ## it will gather all the three component data for the stations if they are available. 
        sample_st = process_station(network, station, location, start_time, end_time, channel_patterns, client)

        if len(sample_st) == 0:
            print(f"No valid data available for station {station}. Skipping.")
            continue

        try:
            # reshaped data now is an array that has undergone linear detrending, and resampling to 100 Hz. 
            # we are trying to do the same processing to this data as was done to the data that neural networks was trained on. 
            # the shape of reshaped data would be (no. of types of sensors in a given station, 3, signal_length)
            reshaped_data = reshape_and_resample(sample_st, signal_length, fs = orig_sr)


            # this is storing the ids. (e.g., NET..STA.BHZ)
            station_ids.append([sample_st[i].id for i in range(len(sample_st))][:3])
            print("Reshaped data:", reshaped_data.shape)


        except ValueError as e:
            print(f"Unable to reshape data for station {station}. Error: {e}")
            continue



        station_wise_outputs = []
        station_wise_probs = []


        # reshaped_data is of the shape (no. of stations/types of sensors present, no. of channels, signal_length)
        for station in reshaped_data:
            windowed_data = []
            
            if model_type == 'dl':
                for i in range(0, int((signal_length*orig_sr) - (window_length * orig_sr)), stride):
                    input_data = station[:, i:i + window_length * orig_sr]

                    # applying the taper and filtering
                    filtered_signal = taper_and_bandpass_filter(input_data, lowcut= lowpass, highcut= highpass, fs= orig_sr)

                    # normalizing by standard deviation as done in the training dataset. 
                    filtered_signal = filtered_signal/np.std(np.abs(filtered_signal))          
                    windowed_data.append(np.array([signal.resample(row, int(window_length*new_sr)) for row in filtered_signal]))

                # spectrogram expects an input of shape (number of windows, number of channels, number of samples)
                spectrograms = extract_spectrograms(waveforms = np.array(windowed_data))

                # this is the input that goes into neural network model. 
                a = torch.Tensor(spectrograms).to(device)

                # if we are testing one-D model
                if one_d:
                    a = torch.Tensor(np.array(windowed_data)).to(device)

                with torch.no_grad():
                    output = model(a)
                    softmax_probs = F.softmax(output, dim=1).cpu().numpy()

                station_wise_outputs.append(output)
                station_wise_probs.append(softmax_probs)       

            
            elif model_type == 'ml':
                
                scaler_params = pd.read_csv('scaler_params_phy_man_'+filename+'.csv')
                best_model = load('best_rf_model_all_features_phy_man_'+filename+'.joblib')
                
                
                if int(filename.split('_')[4]) != 1:
                    lowpass = int(filename.split('_')[4])/10
                    
                else:
                    lowpass = int(filename.split('_')[4])
                    
                highpass = int(filename.split('_')[5])
                
                window_length = int(filename.split('_')[1])+int(filename.split('_')[2])
                
                for i in range(0, int((signal_length*orig_sr) - (window_length * orig_sr)), stride):
                    
                    input_data = np.array([station[2, i:i + window_length * orig_sr]])
                    tapered = apply_cosine_taper(input_data, 10)      
                    filtered_data = np.array(butterworth_filter(tapered, lowpass, highpass, orig_sr, 4, 'bandpass'))
                    normalized_data = filtered_data / np.max(abs(filtered_data), axis=1)[:, np.newaxis]
                    resampled_data = np.array([resample_array(arr, orig_sr, new_sr) for arr in normalized_data])[0]
                    windowed_data.append(resampled_data)
                
                for i in range(len(windowed_data)):
                    try:
                        columns = scaler_params['Feature'].values

                        scaler_params.index = columns

                        physical_features = seis_feature.FeatureCalculator(windowed_data[i], fs=new_sr, selected_features = columns).compute_features()
                        final_features =  physical_features            #pd.concat([tsfel_features, physical_features], axis=1)
                        #columns = scaler_params['Feature'].values
                        features = final_features.loc[:, columns]

                        # Scale features
                        for j in range(len(columns)):
                            features[columns[j]] = (features[columns[j]] - scaler_params.loc[columns[j],'Mean'])/scaler_params.loc[columns[j], 'Std Dev']

                        # Add time features
                        if len(columns) < 30:
                            features['hod'] = (start_time).hour - 8
                        else:
                            features['hod'] = (start_time).hour - 8
                            features['dow'] = (start_time).weekday
                            features['moy'] = (start_time).month

                        # Predict results
                        #best_model.predict(features)
                        station_wise_probs.append(best_model.predict_proba(features))
                    except:
                        station_wise_probs.append(np.array([[0, 0, 0, 0]]))
            
            
            

        big_station_wise_probs.append(station_wise_probs)
        big_reshaped_data.append(reshaped_data)
        big_station_ids.append(station_ids)

        
    return big_station_wise_probs, big_reshaped_data, big_station_ids






def plot_z_component_with_probs(z_data, fs, window_size, softmax_probs, station_names, lowpass = lowpass, highpass = highpass, signal_length = signal_length,  model_name = 'ML P_10_30_F_05_15_50'):
    """
    Plot Z component data with softmax probabilities.
    
    input arguments
    z_data: is the Z component data of all the available stations, it will have a shape of (n_stns, signal_length*orig_sr)
    (note that z_data is sampled in original sampling rate)
    
    fs = sampling rate in which the data is sampled originally
    
    window_size = size of the window for which model is computing the results. 
    
    station_names = the stations ids of each of the participating station. 
    
    softmax_probs: after squeezing, it should be of the shape (n_stns, n_windows, n_classes)
    lowpass: lower frequency limit of a bandpass filter for the display purpose
    highpass: higher frequency limit of a bandpass filter for the display purpose
    
    
    
    """

    softmax_probs = np.squeeze(softmax_probs)
    
    lowcut, highcut = lowpass, highpass
    nyquist = 0.5 * fs
    b, a = signal.butter(4, [lowcut / nyquist, highcut / nyquist], btype='band')
    
    # filtering the Z data for display purpose
    z_data = signal.filtfilt(b, a, z_data / np.max(np.abs(z_data), axis=1, keepdims=True))
    
    # normalizing the filtered Z data for display  purpose. 
    z_data = z_data / np.max(np.abs(z_data), axis=1, keepdims=True)
    
    # defining the time variable. 
    time = np.linspace(0, z_data.shape[-1] / fs, z_data.shape[-1])

    fig, ax = plt.subplots(nrows=len(z_data)+1, ncols=1, figsize=[10, 1.5 * len(z_data)+ 1.5], sharex=True)
    ax = [ax] if len(z_data) == 1 else ax

    
    for i, station_data in enumerate(z_data):
        ax[i].plot(time, station_data, label="Z Component")

        points = np.linspace(0, int((signal_length) - (window_size)), max(softmax_probs[i].shape)) + window_size / 2

        eq_probs, exp_probs, no_probs, su_probs = np.squeeze(softmax_probs[i].T)
   
  
        ax[i].scatter(points, eq_probs, c='k', label='Prob(Eq)', zorder=1, ec= 'k')
        ax[i].scatter(points, exp_probs, c='blue', label='Prob(Exp)', zorder=1, ec = 'k')
        ax[i].scatter(points, su_probs, c='red', label='Prob(Su)', zorder=1, ec = 'k')
        ax[i].scatter(points, no_probs, c='white', label='Prob(No)', zorder=1, ec = 'k')

        ax[i].set_title(f"Station ID: {station_names[i]}")
        ax[i].set_ylabel("Amplitude")
        ax[i].set_xlim(0, int((signal_length)))
    
    
        ax[-1].set_title("Mean Probabilities")
        ax[-1].scatter(points, np.mean(softmax_probs, axis = 0)[:,0],  c='k',  zorder=1, ec= 'k')
        ax[-1].scatter(points, np.mean(softmax_probs, axis = 0)[:,1],  c='blue',  zorder=1, ec= 'k')
        ax[-1].scatter(points, np.mean(softmax_probs, axis = 0)[:,2],  c='white',  zorder=1, ec= 'k')
        ax[-1].scatter(points, np.mean(softmax_probs, axis = 0)[:,3],  c= 'red',  zorder=1, ec= 'k')
 
        
            # Create custom legend for circular markers
        legend_elements = [
           mlines.Line2D([], [], marker='o', color='red', mec = 'k', label='Prob (Su)', markersize= 8),
            mlines.Line2D([], [], marker='o', color='k', mec = 'k', label='Prob (Eq)', markersize= 8), 
            mlines.Line2D([], [], marker='o', color='white', mec = 'k',  label='Prob (No)', markersize= 8),
             mlines.Line2D([], [], marker='o', color='blue', mec = 'k',  label='Prob (Exp)', markersize= 8)
        ]
        ax[-1].legend(handles=legend_elements, loc='upper right', fontsize= 8)

    plt.xlabel("Time (s)")
    fig.suptitle(model_name)
  
    plt.tight_layout()
    plt.show()
    
    
    

def plot_all_model_predictions(fig_size, signal_length, big_reshaped_data, stn_probs_dl, stn_probs_ml, big_station_ids, start_time, orig_sr):
    # Create subplots
    fig, ax = plt.subplots(2, 4, figsize=fig_size, sharey=True, sharex=False)

    # Process data
    data = np.squeeze(np.array(big_reshaped_data))[:, 2, :]
    data = np.array(butterworth_filter(data, 0.5, 15, orig_sr, 4, 'bandpass'))

    # Function to plot scatter points
    def plot_scatter(ax, points, probs, offset, colors):
        eq_probs, exp_probs, no_probs, su_probs = probs.T
        ax.scatter(points, eq_probs + offset, c=colors[0], ec='k')
        ax.scatter(points, exp_probs + offset, c=colors[1], ec='k')
        ax.scatter(points, no_probs + offset, c=colors[2], ec='k')
        ax.scatter(points, su_probs + offset, c=colors[3], ec='k')

    # Iterate through the data and plot DL model results
    for i in range(len(data)):
        time = np.linspace(0, signal_length, len(data[i]))

        for col, (model_probs, model_title) in enumerate(stn_probs_dl):
            ax[0, col].plot(time, (data[i] / np.max(abs(data[i]))) + 2 * i, c='#1f77b4')
            sample_probs = np.squeeze(np.array(model_probs))[i]
            points = np.linspace(0, signal_length - 100, sample_probs.shape[0]) + 50  # Window size of 100

            # Plot individual sample probabilities
            plot_scatter(ax[0, col], points, sample_probs, 2 * i, ['k', 'b', 'white', 'r'])

            # Plot averaged probabilities
            avg_probs = np.mean(np.squeeze(np.array(model_probs)), axis=0)
            plot_scatter(ax[0, col], points, avg_probs, -2, ['k', 'b', 'white', 'r'])

            ax[0, col].set_xlim(0, signal_length)

            # Set title with max probabilities
            max_eq_prob, max_exp_prob, _, max_su_prob = np.round(np.max(np.mean(np.squeeze(model_probs), axis=0), axis=0).astype('float'), 1)
            ax[0, col].set_title(f'{model_title}, Max Eq:{max_eq_prob}, Max Exp:{max_exp_prob}, Max Su:{max_su_prob}')

    # Plot ML model results
    for row, (ml_probs, ml_title, window_size) in enumerate(stn_probs_ml):
        sample_probs = np.squeeze(np.array(ml_probs))

        for i in range(len(data)):
            points = np.linspace(0, signal_length - window_size, sample_probs[i].shape[0]) + window_size / 2
            ax[1, row].plot(time, (data[i] / np.max(abs(data[i]))) + 2 * i, c='#1f77b4')
            plot_scatter(ax[1, row], points, sample_probs[i], 2 * i, ['k', 'b', 'white', 'r'])

            avg_probs = np.mean(sample_probs, axis=0)
            plot_scatter(ax[1, row], points, avg_probs, -2, ['k', 'b', 'white', 'r'])

            ax[1, row].set_xlim(0, signal_length)

        max_eq_prob, max_exp_prob, _, max_su_prob = np.round(np.max(np.mean(np.squeeze(ml_probs), axis=0), axis=0).astype('float'), 1)
        ax[1, row].set_title(f'{ml_title}, Max Eq:{max_eq_prob}, Max Exp:{max_exp_prob}, Max Su:{max_su_prob}')

    # Hide last subplot (1, 3)
    ax[1, 3].axis('off')

    # Set station labels and layout adjustments
    plt.yticks(np.arange(0, 2 * len(data), 2), np.squeeze(big_station_ids)[:, 2])
    fig.suptitle(str(start_time), y=0.99)
    fig.tight_layout()
    plt.show()
    

    
    



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from utils import apply_cosine_taper, butterworth_filter, resample_array

# Function to plot scatter points
def plot_scatter(ax, points, probs, offset, colors):
    eq_probs, exp_probs, no_probs, su_probs = probs.T
    ax.scatter(points, eq_probs + offset, c=colors[0], ec='k')
    ax.scatter(points, exp_probs + offset, c=colors[1], ec='k')
    ax.scatter(points, no_probs + offset, c=colors[2], ec='k')
    ax.scatter(points, su_probs + offset, c=colors[3], ec='k')

# Main function to plot probabilities
def plot_all_model_probs(stn_probs_dl, stn_probs_ml, big_reshaped_data, orig_sr, start_time, end_time, big_station_ids, fig_size = (20, 15)):
    signal_length = end_time - start_time
    fig, ax = plt.subplots(2, 3, figsize=fig_size, sharey=True, sharex=False)

    legend_elements = [
        mlines.Line2D([], [], marker='o', color='#FF0000', mec='k', label='Prob (Eq)', markersize=8),
        mlines.Line2D([], [], marker='o', color='green', mec='k', label='Prob (Exp)', markersize=8),
        mlines.Line2D([], [], marker='o', color='blue', mec='k', label='Prob (Su)', markersize=8)
    ]

    data = np.array(big_reshaped_data)[:, 0, 2, :]
    data = np.array(butterworth_filter(data, 0.5, 15, orig_sr, 4, 'bandpass'))

    for i in range(len(data)):
        time = np.linspace(0, signal_length, len(data[i]))

        for col, (model_probs, model_title) in enumerate(stn_probs_dl):
            ax[0, col].plot(time, (data[i] / np.max(abs(data[i]))) + 2 * i, c='k', lw=0.5, zorder=1, alpha=0.5)
            model_probs = np.array(model_probs).reshape(len(data), -1, 4)
            sample_probs = model_probs[i]
            points = np.linspace(0, signal_length - 100, sample_probs.shape[0]) + 50
            plot_scatter(ax[0, col], points, sample_probs, 2 * i, ['#FF0000', 'g', 'white', 'b'])
            avg_probs = np.mean(model_probs, axis=0)
            plot_scatter(ax[0, col], points, avg_probs, -2, ['#FF0000', 'g', 'white', 'b'])
            ax[0, col].set_xlim(0, signal_length)
            max_eq_prob, max_exp_prob, _, max_su_prob = np.round(np.max(np.mean(model_probs, axis=0), axis=0).astype('float'), 1)
            ax[0, col].set_title(f'{model_title}, Max Eq:{max_eq_prob}, Max Exp:{max_exp_prob}, Max Su:{max_su_prob}')

        for row, (ml_probs, ml_title, window_size) in enumerate(stn_probs_ml):
            sample_probs = np.squeeze(np.array(ml_probs))
            points = np.linspace(0, signal_length - window_size, sample_probs[i].shape[0]) + window_size / 2
            ax[1, row].plot(time, (data[i] / np.max(abs(data[i]))) + 2 * i, c='k', lw=0.5, zorder=1, alpha=0.5)
            model_probs = np.array(ml_probs).reshape(len(data), -1, 4)
            sample_probs = model_probs[i]
            plot_scatter(ax[1, row], points, sample_probs, 2 * i, ['#FF0000', 'g', 'white', 'b'])
            avg_probs = np.mean(model_probs, axis=0)
            plot_scatter(ax[1, row], points, avg_probs, -2, ['#FF0000', 'g', 'white', 'b'])
            ax[1, row].set_xlim(0, signal_length)
            max_eq_prob, max_exp_prob, _, max_su_prob = np.round(np.max(np.mean(model_probs, axis=0), axis=0).astype('float'), 1)
            ax[1, row].set_title(f'{ml_title}, Max Eq:{max_eq_prob}, Max Exp:{max_exp_prob}, Max Su:{max_su_prob}')

    ax[-1, -1].legend(handles=legend_elements, loc='upper left', fontsize=8)
    plt.yticks(np.arange(0, 2 * len(data), 2), np.array(big_station_ids).reshape(len(data), 3)[:, 2])
    fig.suptitle('time since '+str(start_time), y=0.99)
    fig.tight_layout()
    plt.show()



# Main function to plot probabilities
def plot_single_model_probs(stn_probs,  big_reshaped_data, orig_sr, start_time, end_time, big_station_ids, fig_size = (10, 15)):
    signal_length = end_time - start_time
    fig, ax = plt.subplots(1, 1, figsize=fig_size, sharey=True, sharex=False)

    legend_elements = [
        mlines.Line2D([], [], marker='o', color='#FF0000', mec='k', label='Prob (Eq)', markersize=8),
        mlines.Line2D([], [], marker='o', color='green', mec='k', label='Prob (Exp)', markersize=8),
        mlines.Line2D([], [], marker='o', color='blue', mec='k', label='Prob (Su)', markersize=8)
    ]

    data = np.array(big_reshaped_data)[:, 0, 2, :]
    data = np.array(butterworth_filter(data, 0.5, 15, orig_sr, 4, 'bandpass'))

    for i in range(len(data)):
        time = np.linspace(0, signal_length, len(data[i]))


        for row, (ml_probs, ml_title, window_size) in enumerate(stn_probs):
            sample_probs = np.squeeze(np.array(ml_probs))
            points = np.linspace(0, signal_length - window_size, sample_probs[i].shape[0]) + window_size / 2
            ax.plot(time, (data[i] / np.max(abs(data[i]))) + 2 * i, c='k', lw=0.5, zorder=1, alpha=0.5)
            model_probs = np.array(ml_probs).reshape(len(data), -1, 4)
            sample_probs = model_probs[i]
            plot_scatter(ax, points, sample_probs, 2 * i, ['#FF0000', 'g', 'white', 'b'])
            avg_probs = np.mean(model_probs, axis=0)
            plot_scatter(ax, points, avg_probs, -2, ['#FF0000', 'g', 'white', 'b'])
            ax.set_xlim(0, signal_length)
            max_eq_prob, max_exp_prob, _, max_su_prob = np.round(np.max(np.mean(model_probs, axis=0), axis=0).astype('float'), 1)
            ax.set_title(f'{ml_title}, Max Eq:{max_eq_prob}, Max Exp:{max_exp_prob}, Max Su:{max_su_prob}')

    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    plt.yticks(np.arange(0, 2 * len(data), 2), np.array(big_station_ids).reshape(len(data), 3)[:, 2])
    fig.suptitle('time since'+str(start_time), y=0.99)
    fig.tight_layout()
    plt.show()

    

