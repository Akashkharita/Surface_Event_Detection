#!/home/ahutko/miniconda3/envs/surface_dl/bin/python

# import std packages
import os
import sys
import random
from time import time

Timport = time()

import warnings # to silence the torch.load warning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import h5py
import obspy
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from scipy import stats, signal
from tqdm import tqdm
from joblib import dump, load

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torchvision.transforms as transforms

#sys.path.append('../src')

# Add custom module path
module_path = os.path.abspath(os.path.join('..', 'src'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import custom utilities and models
import seis_feature
from utils import apply_cosine_taper, butterworth_filter, resample_array
from neural_network_architectures import (
    QuakeXNet_1d, QuakeXNet_2d, SeismicCNN_1d, SeismicCNN_2d
)

print("DONE IMPORTING.  Elapsed time: ",time()-Timport)
Tzero = time()

# import third party packages
from obspy.clients.fdsn import Client
client = Client('IRIS')
import obspy

# import event classifier specific packges
from db.get_event_info import unix_to_true_time
from db.get_event_info import get_event_info

# Import classification functions
from all_models_classification import (
    compute_window_probs, plot_single_model_probs, plot_all_model_probs
)

#----- Download .joblib models and scaler_params*.csv files. Only do this once.
#doi = '10.5281/zenodo.13334838'
#from zenodo_get import zenodo_get
#files = zenodo_get([doi])

#----- get evid from input arguments
try:
    evid = sys.argv[1]
#    model = sys.argv[2]
except:
    print("Usage: thisscript.py evid") # [40, 110, 150 or 20]")
    print("evid must be a valid int event_id") #and model abbreviation")
    sys.exit(1)

# ====================
# 0. Parameters
# ====================

# Set device to GPU if available, else use CPU
device = "cpu"     #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define parameters for signal processing
orig_sr = 100  # Original sampling rate in Hz
new_sr = 50    # New sampling rate in Hz
stride = 10 * orig_sr  # Window moving increment for processing windows (X seconds times original sampling rate)
lowpass = 1  # Lowpass filter cutoff frequency in Hz
highpass = 20  # Highpass filter cutoff frequency in Hz
window_length = 100  # Length of the window for processing in samples
channel_patterns = ["HH", "BH", "EH", "HN", "EN"]  # Channel patterns to filter
trace_window_length = 300  # Total length of data downloaded

# Get params from evid
orid, ordate, lat, lon, dep, mag, mindist, maxdist, netstas, dists_km, analyst_class = get_event_info(evid)
event_info = [ evid, orid, ordate, lat, lon, dep, mag, mindist, maxdist, netstas, dists_km, analyst_class ]
start_time = UTCDateTime(ordate) - (stride/orig_sr) - window_length
dists_km = [round(x*10)/10. for x in dists_km]
mindist = min(dists_km)
if mindist > 50:
    P_offset = mindist/6. # rough estimate of earliest arrival w.r.t. origin time
    start_time = start_time + P_offset

end_time = start_time + trace_window_length
strordate = ordate.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4]
strordate2 = ordate.strftime("%Y_%m_%dT%H_%M_%S")
stations_id = netstas



#start_time = obspy.UTCDateTime(2025, 3, 26, 22, 48, 22.6)
#end_time = obspy.UTCDateTime(2025, 3, 26, 22, 53, 22.6)




print('')
print("-------------- ", evid, " --------------")
print("ORDATE: ",start_time, end_time, ordate )
print("STATIONS_ID: ",stations_id)

# Location wildcard, used to match all available locations for the given stations
location = "*"


## Define start and end times for the seismic event
#start_time = obspy.UTCDateTime(2024, 8, 15, 17, 39, 52)  # UTC start time
#end_time = start_time + 300  # End time, 300 seconds (5 minutes) after start time

# Initialize client for accessing IRIS web services
client = Client("IRIS")

# Function to process and compute probabilities for seismic models
def process_model(model, stations_id, location, start_time, end_time, one_d, model_type, filename, remember_st=True, st_all=None, integrate_SM=False):
    """
    This function computes the probabilities for a given model, station data, and time window.
    
    Args:
        model: The model used for classification (e.g., deep learning or machine learning).
        stations_id: List of station IDs for data collection.
        st_all: (don
        location: Location of the station (wildcard "*" to match all locations).
        start_time: Start time of the seismic event (obspy.UTCDateTime object).
        end_time: End time of the seismic event (obspy.UTCDateTime object).
        one_d: Boolean indicating whether the model is 1D or 2D.
        model_type: Type of the model, either 'dl' for deep learning or 'ml' for machine learning.
        filename: Name of the output file to save results.
        remember_st (bool, optional): If True, retains the cumulative st_all stream to only download data once per station.
        st_all (Stream, optional): Existing cumulative stream; if None, a new stream is created.
        integrate_SM (optional, default = False): integrate SM data to velocity (with pre-highpass filter above 0.3Hz).

    Returns:
        tuple:
            Processed probabilities and output data.
            st_all (Stream): Updated cumulative waveform stream.
    """
    if st_all is None or not remember_st:
        st_all = Stream()
    return compute_window_probs(
        stations_id=stations_id, st_all=st_all, location=location, start_time=start_time,
        end_time=end_time, channel_patterns=channel_patterns, client=client,
        stride=stride, orig_sr=orig_sr, new_sr=new_sr, window_length=window_length,
        lowpass=lowpass, highpass=highpass, one_d=one_d, model=model,
        model_type=model_type, filename=filename, remember_st = remember_st, 
        integrate_SM = integrate_SM
    )

#----- Loading the machine learning models
# ====================
# 1. Model Setup Parameters
# ====================
# Fixed parameters for the models (do not change)
num_channels = 3        # Number of input channels (seismic data)
dropout = 0.9           # Dropout rate to prevent overfitting

# ============================
# 2. Model Initialization
# ============================
# Initialize models with the number of classes, channels, and dropout rate
model_seismiccnn_1d = SeismicCNN_1d(num_classes=4, num_channels=num_channels, dropout_rate=dropout).to(device)
model_seismiccnn_2d = SeismicCNN_2d(num_classes=4, num_channels=num_channels, dropout_rate=dropout).to(device)
model_quakexnet_1d = QuakeXNet_1d(num_classes=4, num_channels=num_channels, dropout_rate=dropout).to(device)
model_quakexnet_2d = QuakeXNet_2d(num_classes=4, num_channels=num_channels, dropout_rate=dropout).to(device)

# ============================
# 3. Load Pretrained Weights
# ============================
# Load the pretrained model state dictionaries from saved files
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`", category=FutureWarning)
saved_model_seismiccnn_2d = torch.load('../trained_deep_learning_models/best_model_SeismicCNN_2d.pth', map_location=device)
saved_model_quakexnet_2d = torch.load('../trained_deep_learning_models/best_model_QuakeXNet_2d.pth', map_location=device)
saved_model_quakexnet_1d = torch.load('../trained_deep_learning_models/best_model_QuakeXNet_1d.pth', map_location=device)
saved_model_seismiccnn_1d = torch.load('../trained_deep_learning_models/best_model_SeismicCNN_1d.pth', map_location=device)

# ============================
# 4. Load Weights into Models
# ============================
# Load the state dictionaries into the corresponding models
model_seismiccnn_1d.load_state_dict(saved_model_seismiccnn_1d)
model_seismiccnn_2d.load_state_dict(saved_model_seismiccnn_2d)
model_quakexnet_1d.load_state_dict(saved_model_quakexnet_1d)
model_quakexnet_2d.load_state_dict(saved_model_quakexnet_2d)

# ============================
# 5. Set Models to Evaluation Mode
# ============================
# Move models to evaluation mode (important for layers like dropout and batch norm)
model_seismiccnn_1d.eval()
model_seismiccnn_2d.eval()
model_quakexnet_1d.eval()
model_quakexnet_2d.eval()

# ============================
# 6. Move Models to Correct Device
# ============================
# Ensure all models are on the correct device (GPU or CPU)
model_seismiccnn_1d.to(device)
model_seismiccnn_2d.to(device)
model_quakexnet_1d.to(device)
model_quakexnet_2d.to(device)

print("DONE LOADING.  Elapsed time: ",time()-Timport)

# Compute probabilities for different deep learning models

Tzero = time()
# 1D model: QuakexNet
stn_probs_quakexnet_1d, _, big_station_ids, st_all, snrs = process_model(
    model_quakexnet_1d, stations_id, location, start_time, end_time,
    one_d=True, model_type='dl', filename='P_10_30_F_05_15_50', 
)
print("ELAPSED TIME (includes data download) 1d QuakexNet: ",(time() - Tzero))
Tzero = time()

prob_stns = np.array(stn_probs_quakexnet_1d)
prob_stns = prob_stns.reshape(len(big_station_ids), -1,4)
for k in range(0,len(prob_stns)):
    eqprob = np.max(prob_stns[k][:,0])
    exprob = np.max(prob_stns[k][:,1])
    noprob = np.max(prob_stns[k][:,2])
    suprob = np.max(prob_stns[k][:,3])
    pvals = [eqprob, exprob, suprob]    
    sorted_pvals = sorted(pvals, reverse=True)
    pdistance = sorted_pvals[0] - sorted_pvals[1]
    print("PROBS quakexnet1d: ",k,big_station_ids[k], eqprob, exprob, noprob, suprob)

#for col, (model_probs, model_title) in enumerate(stn_probs_quakexnet_1d):
#            model_probs = np.array(model_probs).reshape(len(data), -1, 4)
#            sample_probs = model_probs[i]
#            avg_probs = np.mean(model_probs, axis=0)
#            print('MODEL PROBS: ", evid, "',model_probs)
#            print("SAMPLE PROBS: ", evid, "", sample_probs)
#            print("avg_probs: ",avg_probs)






# 2D model: QuakexNet
stn_probs_quakexnet_2d, big_reshaped_data, big_station_ids, st_all, _  = process_model(
    model_quakexnet_2d, stations_id, location, start_time, end_time,
    one_d=False, model_type='dl', filename='P_10_30_F_05_15_50',
    st_all=st_all
)
print("ELAPSED TIME 2d QuakexNet: ",(time() - Tzero))
Tzero = time()
print('')
prob_stns = np.array(stn_probs_quakexnet_2d)
prob_stns = prob_stns.reshape(len(big_station_ids), -1,4)
for k in range(0,len(prob_stns)):
    eqprob = np.max(prob_stns[k][:,0])
    exprob = np.max(prob_stns[k][:,1])
    noprob = np.max(prob_stns[k][:,2])
    suprob = np.max(prob_stns[k][:,3])
    pvals = [eqprob, exprob, suprob]    
    sorted_pvals = sorted(pvals, reverse=True)
    pdistance = sorted_pvals[0] - sorted_pvals[1]
    print("PROBS quakexnet2d: ",k,big_station_ids[k], eqprob, exprob, noprob, suprob)


# 1D model: SeismicCNN
stn_probs_seismiccnn_1d, _, big_station_ids, st_all, _  = process_model(
    model_seismiccnn_1d, stations_id, location, start_time, end_time,
    one_d=True, model_type='dl', filename='P_10_30_F_05_15_50',
    st_all=st_all
)
print("ELAPSED TIME 1d SeismicCNN: ",(time() - Tzero))
Tzero = time()
print('')
prob_stns = np.array(stn_probs_seismiccnn_1d) 
prob_stns = prob_stns.reshape(len(big_station_ids), -1,4)
for k in range(0,len(prob_stns)):
    eqprob = np.max(prob_stns[k][:,0])
    exprob = np.max(prob_stns[k][:,1])
    noprob = np.max(prob_stns[k][:,2])
    suprob = np.max(prob_stns[k][:,3])
    pvals = [eqprob, exprob, suprob]    
    sorted_pvals = sorted(pvals, reverse=True)
    pdistance = sorted_pvals[0] - sorted_pvals[1]
    print("PROBS: ", evid, "SeismicCNN1d ",k,big_station_ids[k], eqprob, exprob, noprob, suprob)

# 2D model: SeismicCNN
stn_probs_seismiccnn_2d, _, big_station_ids, st_all, _ = process_model(
    model_seismiccnn_2d, stations_id, location, start_time, end_time,
    one_d=False, model_type='dl', filename='P_10_30_F_05_15_50',
    st_all=st_all
)
print("ELAPSED TIME 2d SeismicCNN: ",(time() - Tzero))
Tzero = time()
print('') 
prob_stns = np.array(stn_probs_seismiccnn_2d)
prob_stns = prob_stns.reshape(len(big_station_ids), -1,4)
for k in range(0,len(prob_stns)):
    eqprob = np.max(prob_stns[k][:,0])
    exprob = np.max(prob_stns[k][:,1])
    noprob = np.max(prob_stns[k][:,2])
    suprob = np.max(prob_stns[k][:,3])
    pvals = [eqprob, exprob, suprob]
    sorted_pvals = sorted(pvals, reverse=True)
    pdistance = sorted_pvals[0] - sorted_pvals[1]
    print("PROBS: ", evid, "SeismicCNN2d ",k,big_station_ids[k], eqprob, exprob, noprob, suprob)


# 40 Hz frequency model (ML)
model = model_quakexnet_1d  # dummy name, not important according to Akash
stn_probs_ml_40, _, big_station_ids, st_all, _ = process_model(
    model, stations_id, location, start_time, end_time,
    one_d=False, model_type='ml', filename='P_10_30_F_05_15_50',
    st_all=st_all
)
print("ELAPSED TIME ML40sec: ",(time() - Tzero))
Tzero = time()
print('')
prob_stns = np.array(stn_probs_ml_40)
prob_stns = prob_stns.reshape(len(big_station_ids), -1,4)
for k in range(0,len(prob_stns)):
    eqprob = np.max(prob_stns[k][:,0])
    exprob = np.max(prob_stns[k][:,1])
    noprob = np.max(prob_stns[k][:,2])
    suprob = np.max(prob_stns[k][:,3])
    pvals = [eqprob, exprob, suprob]
    sorted_pvals = sorted(pvals, reverse=True)
    pdistance = sorted_pvals[0] - sorted_pvals[1]
    print("PROBS: ", evid, "ML40sec ",k,big_station_ids[k], eqprob, exprob, noprob, suprob)



def print_stats(probs, name, evid, snrs, analyst_class):
    """
    For the event classes EQ, EX, and SU, calculate and print:
      - The overall mean probability of each event class.
      - The mean probability for each event class, computed over only those stations
        where the maximum probability for that station is greater than a threshold
        AND the probability distance (difference between the top two probabilities)
        is greater than a given threshold.
        
    The output name for each combination is formatted as:
         myname_p<probthreshold>_d<probdistance>
    with the threshold values printed to one decimal place.
    
    Parameters:
      probs (array-like): Probability values that can be reshaped to 
                          (number_of_stations, n_windows, 4).
      name (str): Base name (e.g., "myname").
    """
    # Convert probabilities to a NumPy array and reshape.
    probs = np.array(probs)
    nstations = len(big_station_ids)  # Ensure big_station_ids is defined globally
    probs = probs.reshape(nstations, -1, 4)
    try:
        if len(analyst_class) == 2:
            pass
    except:
        analyst_class = 'na'
    
    # Initialize lists to hold the maximum probability values per station.
    eqprobs, exprobs, suprobs, pdistances = [], [], [], []
    
    # Loop over stations and compute the maximum probability for each event class.
    for k in range(nstations):
        # For each station, assume:
        # index 0: EQ, index 1: EX, index 3: SU.
        eqprob = np.max(probs[k][:, 0])
        exprob = np.max(probs[k][:, 1])
        suprob = np.max(probs[k][:, 3])
        eqprobs.append(eqprob)
        exprobs.append(exprob)
        suprobs.append(suprob)
        
        # Compute probability distance: difference between the highest and second-highest
        pvals = [eqprob, exprob, suprob]
        sorted_pvals = sorted(pvals, reverse=True)
        pdistance = sorted_pvals[0] - sorted_pvals[1]
        pdistances.append(pdistance)
    
    # Convert lists to NumPy arrays for easier indexing.
    eqprobs = np.array(eqprobs)
    exprobs = np.array(exprobs)
    suprobs = np.array(suprobs)
    pdistances = np.array(pdistances)
    snrs = np.array(snrs)

    # Print overall means for each event class.
    overall_eq = np.mean(eqprobs)
    overall_ex = np.mean(exprobs)
    overall_su = np.mean(suprobs)
    overall_snr = np.mean(snrs)
    print(f"{str(evid):<10s}  {name:20s} Overall: EQ: {overall_eq:5.3f}  EX: {overall_ex:5.3f}  SU: {overall_su:5.3f}  SNR: {overall_snr:6.1f}")
   
    # Loop over the desired probability thresholds, probability distance thresholds, and SNR thresholds.
    for probthreshold in [ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99 ]:
        for probdistance in [ 0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4 ]:
            for snrthreshold in [ 1, 2, 3, 4, 10 ]:
                # Create separate masks for each event class:
                eq_mask = (eqprobs > probthreshold) & (pdistances > probdistance) & (snrs > snrthreshold)
                ex_mask = (exprobs > probthreshold) & (pdistances > probdistance) & (snrs > snrthreshold)
                su_mask = (suprobs > probthreshold) & (pdistances > probdistance) & (snrs > snrthreshold)
                
                # Calculate the means for each event class using its own mask.
                mean_eq = np.mean(eqprobs[eq_mask]) if np.any(eq_mask) else 0.0
                mean_ex = np.mean(exprobs[ex_mask]) if np.any(ex_mask) else 0.0
                mean_su = np.mean(suprobs[su_mask]) if np.any(su_mask) else 0.0
                
                # Calculate counts for each event class.
                count_eq = np.sum(eq_mask)
                count_ex = np.sum(ex_mask)
                count_su = np.sum(su_mask)
                
                # Compute the probability distance based on the means:
                # This is the difference between the highest and second-highest mean probability.
                pvals_mean = [mean_eq, mean_ex, mean_su]
                sorted_means = sorted(pvals_mean, reverse=True)
                mean_pd = sorted_means[0] - sorted_means[1]
                
                # Determine the event class with the highest mean.
                event_classes = ['EQ', 'EX', 'SU']
                max_idx = np.argmax(pvals_mean)
                max_class = event_classes[max_idx]
                counts = [count_eq, count_ex, count_su]
                pred_count = counts[max_idx]
                if sorted_means[0] < 0.01 or pred_count == 0:
                    max_class = 'NO'
                
                # Create the composite name including SNR threshold and left-justify in a 25-character field.
                composite_name = f"{name}_p{probthreshold:.2f}_d{probdistance:.2f}_snr{snrthreshold:02d}"
                # Print the event id (evid), composite name, the means, counts, probability distance, and predicted class.
                print(f"{str(evid):<10s} {composite_name:<25s}    EQ: {mean_eq:5.3f}  {count_eq:<3d}  EX: {mean_ex:5.3f}  {count_ex:<3d}  SU: {mean_su:5.3f}  {count_su:<3d}    ProbDist: {mean_pd:5.3f}    Pred: {max_class}  {sorted_means[0]:5.3f}  {pred_count:<3d}  Analyst: {analyst_class}")


print('')
print('')
print('----------------------------------------------')
print('')

prob_stns = np.array(stn_probs_seismiccnn_1d)
prob_stns = prob_stns.reshape(len(big_station_ids), -1,4)
for k in range(0,len(prob_stns)):
    eqprob = np.max(prob_stns[k][:,0])
    exprob = np.max(prob_stns[k][:,1])
    noprob = np.max(prob_stns[k][:,2])
    suprob = np.max(prob_stns[k][:,3])
    pvals = [eqprob, exprob, suprob]
    sorted_pvals = sorted(pvals, reverse=True)
    pdistance = sorted_pvals[0] - sorted_pvals[1]
    print("PROBS: ", evid, "SeismicCNN1d ",k,big_station_ids[k], eqprob, exprob, noprob, suprob)
print_stats(stn_probs_seismiccnn_1d, 'SeismicCNN1d', evid, snrs, analyst_class)

print('')
prob_stns = np.array(stn_probs_seismiccnn_2d)
prob_stns = prob_stns.reshape(len(big_station_ids), -1,4)
for k in range(0,len(prob_stns)):
    eqprob = np.max(prob_stns[k][:,0])
    exprob = np.max(prob_stns[k][:,1])
    noprob = np.max(prob_stns[k][:,2])
    suprob = np.max(prob_stns[k][:,3])
    pvals = [eqprob, exprob, suprob]
    sorted_pvals = sorted(pvals, reverse=True)
    pdistance = sorted_pvals[0] - sorted_pvals[1]
    print("PROBS: ", evid, "SeismicCNN2d ",k,big_station_ids[k], eqprob, exprob, noprob, suprob)
print_stats(stn_probs_seismiccnn_1d, 'SeismicCNN2d', evid, snrs, analyst_class)

print('')
prob_stns = np.array(stn_probs_quakexnet_1d)
prob_stns = prob_stns.reshape(len(big_station_ids), -1,4)
for k in range(0,len(prob_stns)):
    eqprob = np.max(prob_stns[k][:,0])
    exprob = np.max(prob_stns[k][:,1])
    noprob = np.max(prob_stns[k][:,2])
    suprob = np.max(prob_stns[k][:,3])
    pvals = [eqprob, exprob, suprob]
    sorted_pvals = sorted(pvals, reverse=True)
    pdistance = sorted_pvals[0] - sorted_pvals[1]
    print("PROBS: ", evid, "QuakeXnet1d ",k,big_station_ids[k], eqprob, exprob, noprob, suprob)
print_stats(stn_probs_quakexnet_1d, 'QuakeXnet1d', evid, snrs, analyst_class)

print('')
prob_stns = np.array(stn_probs_quakexnet_2d)
prob_stns = prob_stns.reshape(len(big_station_ids), -1,4)
for k in range(0,len(prob_stns)):
    eqprob = np.max(prob_stns[k][:,0])
    exprob = np.max(prob_stns[k][:,1])
    noprob = np.max(prob_stns[k][:,2])
    suprob = np.max(prob_stns[k][:,3])
    pvals = [eqprob, exprob, suprob]
    sorted_pvals = sorted(pvals, reverse=True)
    pdistance = sorted_pvals[0] - sorted_pvals[1]
    print("PROBS: ", evid, "QuakeXnet2d ",k,big_station_ids[k], eqprob, exprob, noprob, suprob)
print_stats(stn_probs_quakexnet_2d, 'QuakeXnet2d', evid, snrs, analyst_class)

print('')
prob_stns = np.array(stn_probs_ml_40)
prob_stns = prob_stns.reshape(len(big_station_ids), -1,4)
for k in range(0,len(prob_stns)):
    eqprob = np.max(prob_stns[k][:,0])
    exprob = np.max(prob_stns[k][:,1])
    noprob = np.max(prob_stns[k][:,2])
    suprob = np.max(prob_stns[k][:,3])
    pvals = [eqprob, exprob, suprob]
    sorted_pvals = sorted(pvals, reverse=True)
    pdistance = sorted_pvals[0] - sorted_pvals[1]
    print("PROBS: ", evid, "QuakeXnet2d ",k,big_station_ids[k], eqprob, exprob, noprob, suprob)
print_stats(stn_probs_quakexnet_2d, 'ML40sec', evid, snrs, analyst_class)



#print(big_station_ids)
#print('SNRs ',snrs)


"""
# 40 Hz frequency model (ML)
stn_probs_ml_40, _, big_station_ids, st_all = process_model(
    model, stations_id, location, start_time, end_time,
    one_d=False, model_type='ml', filename='P_10_30_F_05_15_50',
    st_all=st_all
)
print("ELAPSED TIME ML 40sec: ",(time() - Tzero))
Tzero = time()
print('') 
prob_stns = np.array(stn_probs_ml_40)
prob_stns = prob_stns.reshape(len(big_station_ids), -1,4)
for k in range(0,len(prob_stns)):
    eqprob = np.max(prob_stns[k][:,0])
    exprob = np.max(prob_stns[k][:,1])
    noprob = np.max(prob_stns[k][:,2])
    suprob = np.max(prob_stns[k][:,3])
    print("PROBS: ", evid, "ML40 ",k,big_station_ids[k], eqprob, exprob, noprob, suprob)
"""


