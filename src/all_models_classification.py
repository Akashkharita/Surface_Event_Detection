import os
import sys
import random
from glob import glob
from datetime import datetime
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import h5py
import obspy
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from scipy import stats, signal
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
from joblib import dump, load

# Add other modules from this repo:
#   Add the path to the dir ../src
module_path = os.path.abspath(os.path.join('..', 'src'))
if module_path not in sys.path:
    sys.path.append(module_path)
import seis_feature

from utils import apply_cosine_taper, butterworth_filter, resample_array

#----------------------------

# Device Configuration
device = torch.device("cpu")  # Set computation device to CPU

# Model Parameters
num_channels = 3  # Number of input channels
dropout = 0.9  # Dropout rate for regularization
model = []  # Placeholder for model storage
one_d = False  # Flag for 1D data processing

# Seismic Data Parameters
client = Client("IRIS")  # Initialize FDSN client (e.g., IRIS)

# Sampling Parameters
orig_sr = 100  # Original sampling rate (Hz)
new_sr = 50  # Resampled rate (Hz)
os = orig_sr  # Alias for original sampling rate
fs = new_sr  # Alias for new sampling rate
stride = 10 * os  # Stride length for windowing
window_length = 100  # Window length for processing

# Filtering Parameters
lowpass = 1  # Bandpass filter low-frequency corner (Hz)
highpass = 20  # Bandpass filter high-frequency corner (Hz)

##---functions--start---here----##


def get_station_inventory(network, station, location, start_time, end_time, client):
    """
    Fetches station inventory data from the FDSN client.

    Parameters:
    - network (str): Seismic network code.
    - station (str): Station code.
    - location (str): Location identifier (wildcard '*' for all locations).
    - start_time (UTCDateTime): Start time for data retrieval.
    - end_time (UTCDateTime): End time for data retrieval.
    - client (Client): FDSN client instance.

    Returns:
    - inventory (Inventory or None): Inventory object if successful, None if an error occurs.
    """
    try:
        inventory = client.get_stations(
            network=network, station=station, location=location,
            channel="*", starttime=start_time, endtime=end_time, level="channel"
        )
        return inventory
    except Exception as e:
        #print(f"Error fetching station data for {station}: {e}")
        return None


def return_three_components(station, prefix):
    """
    For a given station and channel prefix, return a set of three channel codes to download.
    
    For non-'EH' prefixes, the expected three channels are:
        {prefix + "E", prefix + "N", prefix + "Z"}
    For the 'EH' prefix, two cases are allowed:
      1. If the station has a full EH set: {"EHE", "EHN", "EHZ"} then return that.
      2. Otherwise, if the station has only an EH vertical (i.e. {"EHZ"}) and 
         either a full set of EN channels {"ENE", "ENN"} is available or a full set 
         of HN channels {"HNE", "HNN"} is available, return the corresponding set:
            - {"EHZ", "ENE", "ENN"}  OR  {"EHZ", "HNE", "HNN"}
    
    If none of these conditions is met, return None.
    
    Parameters:
        station (Station): A station object from the inventory.
        prefix (str): A channel prefix (e.g., 'HH', 'BH', or 'EH').
        
    Returns:
        set or None: A set of channel codes to request (or None if not available).
    """
    if prefix != 'EH':
        channels = {chan.code for chan in station.channels if chan.code.startswith(prefix)}
        required = {f"{prefix}E", f"{prefix}N", f"{prefix}Z"}
        if required.issubset(channels):
            return required
        else:
            return None
    else:
        # For EH, first try the standard EH set.
        channels_eh = {chan.code for chan in station.channels if chan.code.startswith('EH')}
        full_eh = {"EHE", "EHN", "EHZ"}
        if full_eh.issubset(channels_eh):
            return full_eh
        # Otherwise, if only EHZ is present from EH, check alternatives.
        if channels_eh == {"EHZ"}:
            channels_en = {chan.code for chan in station.channels if chan.code.startswith('EN')}
            channels_hn = {chan.code for chan in station.channels if chan.code.startswith('HN')}
            if {"ENE", "ENN"}.issubset(channels_en):
                return {"ENE", "ENN", "EHZ"}
            elif {"EN1", "EN2"}.issubset(channels_en):
                return {"EN1", "EN2", "EHZ"}
            elif {"HNE", "HNN"}.issubset(channels_hn):
                return {"HNE", "HNN", "EHZ"}
            elif {"HN1", "HN2"}.issubset(channels_hn):
                return {"HN1", "HN2", "EHZ"}
            else:
                return {"EHZ", "EHZ", "EHZ"}
        return None


def fetch_waveform_data(client, network, station, location, channels_str, start_time, end_time):
    """
    Retrieves waveform data from the FDSN client for given channel codes.

    Parameters:
        client (Client): FDSN client instance.
        network (str): Seismic network code.
        station (str): Station code.
        location (str): Location identifier.
        channels_str (str): Channel codes as a comma-separated string.
        start_time (UTCDateTime): Start time for data retrieval.
        end_time (UTCDateTime): End time for data retrieval.

    Returns:
        Stream: Waveform data stream (empty if retrieval fails).
    """
    try:
        textra = 0.05  # small buffer for IRIS slicing issues
        st = client.get_waveforms(
            network=network, station=station, location=location,
            channel=channels_str,
            starttime=start_time - textra, endtime=end_time,
            minimumlength=end_time - start_time - textra
        )
        st.trim(start_time, end_time)
        return st
    except Exception as e:
        print(f"Error fetching waveform data for {station}: {e}")
        from obspy import Stream
        return Stream()


from obspy import Stream

def sort_stream_by_channel_component(st):
    """
    Sort an ObsPy Stream (with three traces) by the last character of each trace's channel code.
    The desired ordering depends on which set of endings is present:
      - If the endings are {'Z', '1', '2'}, then the order should be:
          Traces ending in '1' come first, then '2', then 'Z'.
      - If the endings are {'1', '2', '3'}, then the order should be:
          Traces ending in '2' come first, then '3', then '1'.
      - If the endings are {'Z', 'N', 'E'}, then the order should be:
          Traces ending in 'E' come first, then 'N', then 'Z'.
    If the endings do not exactly match any of these sets, the function does not resort.
    Parameters:
        st (obspy.Stream): An ObsPy Stream containing three traces.
        
    Returns:
        obspy.Stream: A new Stream with traces sorted according to the specified ordering.
    """
    # Extract the set of last characters from each trace's channel code.
    endings = {tr.stats.channel[-1] for tr in st}
    
    # Determine the ordering mapping based on the endings present.
    if endings == {'Z', '1', '2'}:
        # Desired order: '1' -> '2' -> 'Z'
        mapping = {'1': 0, '2': 1, 'Z': 2}
    elif endings == {'1', '2', '3'}:
        # Desired order: '2' -> '3' -> '1'
        mapping = {'2': 0, '3': 1, '1': 2}
    elif endings == {'Z', 'N', 'E'}:
        # Desired order: 'E' -> 'N' -> 'Z'
        mapping = {'E': 0, 'N': 1, 'Z': 2}
    else:
        return(st)

    # Sort the stream. For each trace, use the mapping value for its channel's last character.
    # If a trace's last character is not in mapping, default to 99 (placing it at the end).
    sorted_traces = sorted(st, key=lambda tr: mapping.get(tr.stats.channel[-1], 99))
    
    # Return a new ObsPy Stream with the sorted traces.
    return Stream(sorted_traces)


def get_waveform_station(network, station, location, start_time, end_time,
                         channel_patterns, client, st_all, plot_data=False, integrate_SM=False):
    """
    Processes a seismic station by fetching available waveform data.

    Parameters:
        network (str): Seismic network code.
        station (str): Station code.
        location (str): Location identifier.
        start_time (UTCDateTime): Start time for data retrieval.
        end_time (UTCDateTime): End time for data retrieval.
        channel_patterns (list): List of channel prefixes to check (e.g., ['HH', 'BH', 'EH']).
        client (Client): FDSN client instance.
        st_all (Stream): Cumulative stream of station data.
        plot_data (bool, optional): If True, plots the retrieved waveforms.
        integrate_SM (bool, optional): If true, integrate SM data to velocity.

    Returns:
        tuple:
            st_station (Stream): Data stream for the station containing all channels.
            st_all (Stream): Updated cumulative stream with station data appended.
    """
    import time
    from obspy import Stream

    # Check if station has already been downloaded
    if len(st_all) > 0 and any(tr.stats.station == station for tr in st_all):
        st_station = st_all.select(station=station)
    else:
        inventory = get_station_inventory(network, station, location, start_time, end_time, client)
        if inventory is None:
            return Stream(), st_all

        print('')
        st_station = Stream()

        # Loop over networks and stations in the inventory.
        for net in inventory:
            for sta in net:
                # Try each prefix in order.
                for prefix in channel_patterns:
                    channels_set = return_three_components(sta, prefix)
                    if channels_set is not None:
                        # Build a channel string by joining the returned codes.
                        # Example: "HHE,HHN,HHZ" or "ENE,ENN,EHZ"
                        channel_str = ",".join(sorted(channels_set))
                        time.sleep(0.05)  # intentional pause to avoid IRIS IP jail due to too many concurrent requests
                        st = fetch_waveform_data(client, net.code, sta.code, location, channel_str, start_time, end_time)
                        # Make sure the stream is channel component ordered: E,N,Z (or 1,2,Z or 1,2,3)
                        if len(st) == 3:
                           st = sort_stream_by_channel_component(st)
                        # Spoof for if EHZ is only trace: have stream = 3 EHZ traces
                        if channels_set == {'EHZ'} and len(st) == 1:
                            st.append(st[0].copy())
                            st.append(st[0].copy())
                        # Do you want to integrate strong motion to velocity? Assumes data later filtered above 1Hz
                        if integrate_SM == True:
                            for tr in st:
                                if tr.stats.channel[1] == 'N':
                                    tr.detrend('demean').detrend('linear').taper(0.01).filter('highpass',freq=0.3).integrate()
                        st_station += st
                        if plot_data:
                            st.plot()
                        break  # Found a valid set, no need to try further prefixes.
                else:
                    # Continue to next station if none of the prefixes returned a valid set.
                    continue
                # Break out of outer loops if data for station has been obtained.
                break
        st_all += st_station
    return st_station, st_all


def taper_and_bandpass_filter(data, lowcut, highcut, fs):
    """
    Applies a bandpass filter and a taper window to the input seismic data.

    Parameters:
    - data (numpy.ndarray): Input waveform data.
    - lowcut (float): Low-pass filter cutoff frequency (Hz).
    - highcut (float): High-pass filter cutoff frequency (Hz).
    - fs (int): Sampling frequency (Hz).

    Returns:
    - numpy.ndarray: Filtered waveform data.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')

    # Apply Tukey window taper
    taper = signal.windows.tukey(data.shape[-1], alpha=0.1)
    data = np.array([np.multiply(taper, row) for row in data])

    # Apply bandpass filter
    data = np.array([signal.filtfilt(b, a, row) for row in data])

    return data


def reshape_and_resample(st, signal_length, fs=100):
    """
    Detrends, resamples, and reshapes seismic waveform data.

    Parameters:
    - st (numpy.ndarray): Input waveform data.
    - signal_length (float): Duration of the signal (seconds).
    - fs (int, optional): Target sampling frequency (Hz). Default is 100 Hz.

    Returns:
    - numpy.ndarray: Resampled and reshaped waveform data.
    """
    st.detrend(type='linear')
    st = np.array([signal.resample(row, int(signal_length * fs)) for row in st])

    reshaped_data = np.array(st).reshape(len(st) // 3, 3, -1)[0].reshape(1, 3, -1)
    return reshaped_data


def extract_spectrograms(waveforms, fs, nperseg=256, overlap=0.5):
    """
    Computes spectrograms from input waveform data.

    Parameters:
    - waveforms (numpy.ndarray): Input waveform data of shape (batch, channels, time).
    - fs (int): Sampling frequency (Hz).
    - nperseg (int, optional): Length of each FFT segment. Default is 256.
    - overlap (float, optional): Fraction of segment overlap. Default is 0.5.

    Returns:
    - numpy.ndarray: Spectrogram data of shape (batch, channels, frequencies, time_segments).
    """
    noverlap = int(nperseg * overlap)  # Compute overlap in samples

    # Compute shape of one spectrogram to initialize array
    f, t, Sxx = signal.spectrogram(waveforms[0, 0], nperseg=nperseg, noverlap=noverlap, fs=fs)

    # Initialize array to store spectrograms
    spectrograms = np.zeros((waveforms.shape[0], waveforms.shape[1], len(f), len(t)))

    # Compute spectrograms for each waveform and channel
    for i in range(waveforms.shape[0]):  # Iterate over waveforms
        for j in range(waveforms.shape[1]):  # Iterate over channels
            _, _, Sxx = signal.spectrogram(waveforms[i, j], nperseg=nperseg, noverlap=noverlap, fs=fs)
            spectrograms[i, j] = Sxx  # Store computed spectrogram

    return spectrograms


def compute_window_probs(
    stations_id, st_all, location, start_time, end_time, channel_patterns, client,
    orig_sr, new_sr, window_length, lowpass, stride, highpass, one_d,
    model, model_type='dl', filename='P_10_30_F_05_15_50', remember_st=True, integrate_SM=False
):
    """
    Computes probability outputs for seismic stations using deep learning (DL) or machine learning (ML) models.

    Parameters:
    - stations_id (list): List of station IDs in 'NET.STA' format.
    - st_all (Stream): Cumulative stream of waveform data. Should be an ObsPy Stream.
    - location (str): Location identifier.
    - start_time (UTCDateTime): Start time for waveform data.
    - end_time (UTCDateTime): End time for waveform data.
    - channel_patterns (list): List of channel prefixes to check (e.g., ['HH', 'BH', 'EH']).
    - client (Client): FDSN client instance.
    - orig_sr (int): Original sampling rate (Hz).
    - new_sr (int): New sampling rate (Hz) after resampling.
    - window_length (int): Window length in seconds.
    - lowpass (float): Bandpass filter low frequency corner (Hz).
    - stride (int): Window moving increment in samples (using original sample rate).
    - highpass (float): Bandpass filter high frequency corner (Hz).
    - one_d (bool): If True, uses a 1D model instead of spectrogram-based input.
    - model (PyTorch model or ML model): Trained model for classification.
    - model_type (str, optional): 'dl' for deep learning, 'ml' for machine learning. Defaults to 'dl'.
    - filename (str, optional): File identifier for model-related parameters. Defaults to 'P_10_30_F_05_15_50'.
    - remember_st (optional, default = True): retain the st_all stream in order to only download data once from a station.
    - integrate_SM (optional, default = False): integrate SM data to velocity (with pre-highpass filter above 0.3Hz).

    Returns:
        tuple:
            big_station_wise_probs (list): Probability outputs for each station.
            big_reshaped_data (list): Reshaped and processed waveform data.
            big_station_ids (list): List of station IDs corresponding to the data.
            st_all (Stream): Updated cumulative waveform stream.
    """
    big_station_wise_probs, big_reshaped_data, big_station_ids = [], [], []
    signal_length = end_time - start_time

    # If we are not retaining previously downloaded data, start fresh.
    if not remember_st or not (isinstance(st_all, Stream) and len(st_all) > 0):
        st_all = Stream()

    for stn_id in stations_id:
        network, station = stn_id.split('.')
        station_ids = []
        sample_st, st_all = get_waveform_station(network, station, location, start_time, end_time, channel_patterns, client, st_all, integrate_SM)
        if len(sample_st) == 0:
            #print(f"No valid & gapless data available for station {station}. Skipping.")
            continue

        try:
            reshaped_data = reshape_and_resample(sample_st, signal_length, fs=orig_sr)
            station_ids.append([sample_st[i].id for i in range(len(sample_st))][:3])
        except ValueError as e:
            print(f"Unable to reshape data for station {station}. Error: {e}")
            continue

        station_wise_outputs, station_wise_probs = [], []

        for station in reshaped_data:
            windowed_data = []
            
            if model_type == 'dl':
                for i in range(0, int((signal_length * orig_sr) - (window_length * orig_sr)), stride):
                    input_data = station[:, i:i + window_length * orig_sr]
                    filtered_signal = taper_and_bandpass_filter(input_data, lowcut=lowpass, highcut=highpass, fs=orig_sr)
                    filtered_signal /= np.std(np.abs(filtered_signal))  # Normalize by standard deviation
                    windowed_data.append(np.array([signal.resample(row, int(window_length * new_sr)) for row in filtered_signal]))
                
                spectrograms = extract_spectrograms(waveforms=np.array(windowed_data), fs = new_sr)
                a = torch.Tensor(spectrograms).to(device)

                if one_d:
                    a = torch.Tensor(np.array(windowed_data)).to(device)

                with torch.no_grad():
                    output = model(a)
                    softmax_probs = F.softmax(output, dim=1).cpu().numpy()

                station_wise_outputs.append(output)
                station_wise_probs.append(softmax_probs)
            
            elif model_type == 'ml':
                scaler_params = pd.read_csv(f'scaler_params_phy_man_{filename}.csv')
                best_model = load(f'best_rf_model_all_features_phy_man_{filename}.joblib')

                lowpass = int(filename.split('_')[4]) / 10 if int(filename.split('_')[4]) != 1 else int(filename.split('_')[4])
                highpass = int(filename.split('_')[5])
                window_length = int(filename.split('_')[1]) + int(filename.split('_')[2])
                
                for i in range(0, int((signal_length * orig_sr) - (window_length * orig_sr)), stride):
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
                        physical_features = seis_feature.FeatureCalculator(windowed_data[i], fs=new_sr, selected_features=columns).compute_features()
                        features = physical_features.loc[:, columns]

                        for j in range(len(columns)):
                            features[columns[j]] = (features[columns[j]] - scaler_params.loc[columns[j],'Mean']) / scaler_params.loc[columns[j], 'Std Dev']
                        
                        if len(columns) < 30:
                            features['hour_of_day'] = start_time.hour - 8
                        else:
                            features['hour_of_day'] = start_time.hour - 8
                            features['day_of_week'] = start_time.weekday
                            features['month_of_year'] = start_time.month
                        
                        station_wise_probs.append(best_model.predict_proba(features))
                    except:
                        station_wise_probs.append(np.array([[0, 0, 0, 0]]))
            
        big_station_wise_probs.append(station_wise_probs)
        big_reshaped_data.append(reshaped_data)
        big_station_ids.append(station_ids)
    
    return big_station_wise_probs, big_reshaped_data, big_station_ids, st_all


# Function to plot scatter points
def plot_scatter(ax, points, probs, offset, colors):
    """
    Plots scatter points for different probability classes.
    
    Parameters:
        ax (matplotlib.axes.Axes): Axis to plot on.
        points (array-like): X-coordinates of scatter points.
        probs (array-like): Probabilities for different classes.
        offset (float): Offset for y-coordinates.
        colors (list): Colors for different probability classes.
    """
    eq_probs, exp_probs, no_probs, su_probs = probs.T
    ax.scatter(points, eq_probs + offset, c=colors[0], ec='k')
    ax.scatter(points, exp_probs + offset, c=colors[1], ec='k')
    ax.scatter(points, no_probs + offset, c=colors[2], ec='k')
    ax.scatter(points, su_probs + offset, c=colors[3], ec='k')

    
# Function to plot probabilities from multiple models
def plot_all_model_probs(stn_probs_dl, stn_probs_ml, big_reshaped_data, orig_sr, start_time, end_time, big_station_ids, fig_size=(20, 15)):
    """
    Plots probability outputs from deep learning and machine learning models.
    
    Parameters:
        stn_probs_dl (list): Deep learning model probabilities.
        stn_probs_ml (list): Machine learning model probabilities.
        big_reshaped_data (array-like): Seismic waveform data.
        orig_sr (float): Original sampling rate.
        start_time (datetime): Start time of the signal.
        end_time (datetime): End time of the signal.
        big_station_ids (list): Station IDs.
        fig_size (tuple): Figure size.
    """
    signal_length = end_time - start_time
    fig, ax = plt.subplots(2, 3, figsize=fig_size, sharey=True, sharex=False)

    legend_elements = [
        mlines.Line2D([], [], marker='o', color='#FF0000', mec='k', label='Prob (Eq)', markersize=8),
        mlines.Line2D([], [], marker='o', color='green', mec='k', label='Prob (Exp)', markersize=8),
        mlines.Line2D([], [], marker='o', color='blue', mec='k', label='Prob (Su)', markersize=8)
    ]

    data = np.array(big_reshaped_data)[:, 0, 2, :]
    data = butterworth_filter(data, 0.5, 15, orig_sr)

    for i in range(len(data)):
        time = np.linspace(0, signal_length, len(data[i]))

        for col, (model_probs, model_title) in enumerate(stn_probs_dl):
            ax[0, col].plot(time, (data[i] / np.max(abs(data[i]))) + 2 * i, c='k', lw=0.5, alpha=0.5)
            model_probs = np.array(model_probs).reshape(len(data), -1, 4)
            sample_probs = model_probs[i]
            points = np.linspace(0, signal_length - 100, sample_probs.shape[0]) + 0
            plot_scatter(ax[0, col], points, sample_probs, 2 * i, ['#FF0000', 'g', 'white', 'b'])
            avg_probs = np.mean(model_probs, axis=0)
            plot_scatter(ax[0, col], points, avg_probs, -2, ['#FF0000', 'g', 'white', 'b'])
            ax[0, col].set_xlim(0, signal_length)
            max_eq_prob, max_exp_prob, _, max_su_prob = np.round(np.max(np.mean(model_probs, axis=0), axis=0).astype('float'), 1)
            ax[0, col].set_title(f'{model_title}, Max Eq:{max_eq_prob}, Max Exp:{max_exp_prob}, Max Su:{max_su_prob}')
            

        for row, (ml_probs, ml_title, window_size) in enumerate(stn_probs_ml):
            sample_probs = np.squeeze(np.array(ml_probs))
            points = np.linspace(0, signal_length - window_size, sample_probs[i].shape[0]) + window_size / 2
            ax[1, row].plot(time, (data[i] / np.max(abs(data[i]))) + 2 * i, c='k', lw=0.5, alpha=0.5)
            model_probs = np.array(ml_probs).reshape(len(data), -1, 4)
            sample_probs = model_probs[i]
            plot_scatter(ax[1, row], points, sample_probs, 2 * i, ['#FF0000', 'g', 'white', 'b'])
            avg_probs = np.mean(model_probs, axis=0)
            plot_scatter(ax[1, row], points, avg_probs, -2, ['#FF0000', 'g', 'white', 'b'])
            ax[1, row].set_xlim(0, signal_length)
            ax[1, row].set_title(f'{ml_title}')
            max_eq_prob, max_exp_prob, _, max_su_prob = np.round(np.max(np.mean(model_probs, axis=0), axis=0).astype('float'), 1)
            ax[1, row].set_title(f'{ml_title}, Max Eq:{max_eq_prob}, Max Exp:{max_exp_prob}, Max Su:{max_su_prob}')

    ax[-1, -1].legend(handles=legend_elements, loc='upper left', fontsize=8)
    plt.yticks(np.arange(0, 2 * len(data), 2), np.array(big_station_ids).reshape(len(data), 3)[:, 2])
    fig.suptitle('Time since ' + str(start_time), y=0.99)
    fig.tight_layout()
    plt.show()


def plot_single_model_probs(stn_probs, big_reshaped_data, orig_sr, start_time, end_time, big_station_ids, fig_size=(10, 15)):
    """
    Plots probability distributions for a single model over time across multiple stations.
    
    Parameters:
    - stn_probs: List of tuples containing probability data, model title, and window size.
    - big_reshaped_data: Numpy array of reshaped seismic waveform data.
    - orig_sr: Original sampling rate of the waveform.
    - start_time: Start time of the signal.
    - end_time: End time of the signal.
    - big_station_ids: List of station IDs corresponding to the data.
    - fig_size: Tuple specifying figure size (default is (10, 15)).
    """
    signal_length = end_time - start_time
    fig, ax = plt.subplots(figsize=fig_size, sharey=True, sharex=False)

    # Define legend elements
    legend_elements = [
        mlines.Line2D([], [], marker='o', color='#FF0000', mec='k', label='Prob (Eq)', markersize=8),
        mlines.Line2D([], [], marker='o', color='green', mec='k', label='Prob (Exp)', markersize=8),
        mlines.Line2D([], [], marker='o', color='blue', mec='k', label='Prob (Su)', markersize=8)
    ]

    # Apply Butterworth filter to the data
    data = np.array(big_reshaped_data)[:, 0, 2, :]
    data = np.array(butterworth_filter(data, 0.5, 15, orig_sr, 4, 'bandpass'))

    for i, waveform in enumerate(data):
        time = np.linspace(0, signal_length, len(waveform))

        for row, (ml_probs, ml_title, window_size) in enumerate(stn_probs):
            sample_probs = np.squeeze(np.array(ml_probs))
            points = np.linspace(0, signal_length - window_size, sample_probs[i].shape[0]) + window_size / 2
            
            # Plot waveform
            ax.plot(time, (waveform / np.max(abs(waveform))) + 2 * i, c='k', lw=0.5, zorder=1, alpha=0.5)
            
            # Process model probabilities
            model_probs = np.array(ml_probs).reshape(len(data), -1, 4)
            sample_probs = model_probs[i]
            plot_scatter(ax, points, sample_probs, 2 * i, ['#FF0000', 'g', 'white', 'b'])
            
            # Compute and plot average probabilities
            avg_probs = np.mean(model_probs, axis=0)
            plot_scatter(ax, points, avg_probs, -2, ['#FF0000', 'g', 'white', 'b'])
            
            # Set axis limits and title
            ax.set_xlim(0, signal_length)
            max_eq_prob, max_exp_prob, _, max_su_prob = np.round(np.max(np.mean(model_probs, axis=0), axis=0).astype('float'), 1)
            ax.set_title(f'{ml_title}, Max Eq: {max_eq_prob}, Max Exp: {max_exp_prob}, Max Su: {max_su_prob}')

    # Add legend and format plot
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    plt.yticks(np.arange(0, 2 * len(data), 2), np.array(big_station_ids).reshape(len(data), 3)[:, 2])
    fig.suptitle(f'Time since {start_time}', y=0.99)
    fig.tight_layout()
    plt.show()

