# Enable autoreload
#%reload_ext autoreload
#%autoreload 2 

import pandas as pd
import numpy as np
import ast
import re


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
from time import time
import pickle
from joblib import dump, load


import os
# Specify the directory containing the module
module_path = os.path.abspath(os.path.join('..', 'src'))

# Add the directory to sys.path
if module_path not in sys.path:
    sys.path.append(module_path)

import seis_feature
from utils import apply_cosine_taper, butterworth_filter, resample_array
 

from neural_network_architectures import QuakeXNet_1d
from neural_network_architectures import QuakeXNet_2d
from neural_network_architectures import SeismicCNN_1d
from neural_network_architectures import SeismicCNN_2d

import json

from zenodo_get import zenodo_get

from all_models_classification import compute_window_probs
from all_models_classification import plot_z_component_with_probs
from all_models_classification import plot_all_model_predictions



doi = '10.5281/zenodo.14884987'
files = zenodo_get([doi])

def return_stations_and_distances(data_string):
    # Step 1: Find the indices of the lists
    stations_start = data_string.index('[')
    stations_end = data_string.index(']') + 1
    distances_start = data_string.index('[', stations_end)
    distances_end = data_string.index(']', distances_start) + 1

    # Step 2: Extract and clean the lists
    stations_list = data_string[stations_start:stations_end].strip("[]").replace("'", "").split(', ')
    distances_list = list(map(int, data_string[distances_start:distances_end].strip("[]").split(', ')))

    return stations_list, distances_list



# Step 1: Read the text file
## the text file contains entries in the format - 
## eventid, , date, time, lat, lon, depth, ml, ,, stns, dist, labels
file_path = '../src/evid_netstas_for_akash.txt'  # Replace with your actual file name
with open(file_path, 'r') as f:
    lines = f.readlines()
    
    

# Step 2: Initialize data list
data = []

# Step 3: Process each line in the file
for line in lines:
    line = line.strip()
    
    try:
        stations_list, distances_list = return_stations_and_distances(line)
        parts = line.split()

        # Extract data
        event_id = parts[0]
        origin_id = parts[1]
        timestamp = f"{parts[2]} {parts[3]}"  # Timestamp (date and time)
        latitude = float(parts[4])
        longitude = float(parts[5])
        depth = float(parts[6])

        # Use regex to extract magnitude type and value
        match = re.match(r'([a-zA-Z]+)(-?[0-9]*\.?[0-9]+)', parts[7])
        magnitude_type = match.group(1)
        magnitude = float(match.group(2))

        min_dist = float(parts[8])
        max_dist = float(parts[9])
        event_type = parts[-1]

        # Append parsed data as a dictionary
        data.append({
            'event_id': event_id,
            'origin_id': origin_id,
            'timestamp': timestamp,
            'event_latitude': latitude,
            'event_longitude': longitude,
            'depth': depth,
            'magnitude_type': magnitude_type,
            'magnitude_value': magnitude,
            'min_distance': min_dist,
            'max_distance': max_dist,
            'stations': stations_list,
            'dist': distances_list,
            'analyst': event_type
        })
        
    except (ValueError, IndexError, AttributeError):
        # Catch only specific errors
        pass

# Step 4: Create a DataFrame
df1 = pd.DataFrame(data)



print(f'Total number of events in the catalog:{len(df1)}')
df1_labelled = df1[df1['analyst'] != 'N/A']

## removing the lf events.
df1_labelled = df1_labelled[df1_labelled['analyst'] != 'lf']

print(f'Total number of analyst labeled events in the catalog:{len(df1_labelled)}')



## setting up some important parameters (not to be changed)
num_channels = 3
dropout = 0.9
# Check if a GPU is available
device = torch.device("cpu")



## initiating the model architectures - 
model_seismiccnn_1d = SeismicCNN_1d(num_classes=4, num_channels=num_channels,dropout_rate=dropout).to(device)  # Use 'cuda' if you have a GPU available
model_seismiccnn_2d = SeismicCNN_2d(num_classes=4, num_channels=num_channels,dropout_rate=dropout).to(device)  # Use 'cuda' if you have a GPU available
model_quakexnet_1d = QuakeXNet_1d(num_classes=4, num_channels=num_channels,dropout_rate=dropout).to(device)  # Use 'cuda' if you have a GPU available
model_quakexnet_2d = QuakeXNet_2d(num_classes=4, num_channels=num_channels,dropout_rate=dropout).to(device)  # Use 'cuda' if you have a GPU available


# Load the saved model state dict (weights)
saved_model_seismiccnn_1d = torch.load('../trained_deep_learning_models/best_model_SeismicCNN_1d.pth', map_location=torch.device('cpu'))  # No'weights_only' argument

# Load the saved model state dict (weights)
saved_model_seismiccnn_2d = torch.load('../trained_deep_learning_models/best_model_SeismicCNN_2d.pth', map_location=torch.device('cpu'))  # No'weights_only' argument

# Load the saved model state dict (weights)
saved_model_quakexnet_1d = torch.load('../trained_deep_learning_models/best_model_QuakeXNet_1d.pth', map_location=torch.device('cpu'))  # No 'weights_only' argument

# Load the saved model state dict (weights)
saved_model_quakexnet_2d = torch.load('../trained_deep_learning_models/best_model_QuakeXNet_2d.pth', map_location=torch.device('cpu'))  # No 'weights_only' argument





# Load the state dict into the model
model_seismiccnn_1d.load_state_dict(saved_model_seismiccnn_1d)
model_seismiccnn_2d.load_state_dict(saved_model_seismiccnn_2d)
model_quakexnet_1d.load_state_dict(saved_model_quakexnet_1d)
model_quakexnet_2d.load_state_dict(saved_model_quakexnet_2d)




# Move the model to the correct device (GPU or CPU)
model_seismiccnn_1d.to(device)
model_seismiccnn_2d.to(device)
model_quakexnet_1d.to(device)
model_quakexnet_2d.to(device)




# Put the model in evaluation mode (important for models with dropout/batch norm layers)
model_seismiccnn_1d.eval()
model_seismiccnn_2d.eval()
model_quakexnet_1d.eval()
model_quakexnet_2d.eval()



client = Client("IRIS")

# Define parameters
orig_sr = 100  # original sampling rate
new_sr = 50    # new sampling rate
stride = 10 * orig_sr
lowpass = 1
highpass = 20
window_length = 100
channel_patterns = ["EH", "BH", "HH"]

# Initialize result lists
probs_seismiccnn_1d, probs_seismiccnn_2d, probs_quakexnet_1d, probs_quakexnet_2d = [], [], [], []
probs_ml_40, probs_ml_110, probs_ml_150 = [], [], []
station_data, station_ids = [], []
evids = []

# Function to process and compute probabilities
def process_model(model, stations_id, location, start_time, end_time, one_d, model_type, filename):
    return compute_window_probs(
        stations_id=stations_id, location=location, start_time=start_time, 
        end_time=end_time, channel_patterns=channel_patterns, client=client, 
        stride=stride, orig_sr=orig_sr, new_sr=new_sr, window_length=window_length, 
        lowpass=lowpass, highpass=highpass, one_d=one_d, model=model, 
        model_type=model_type, filename=filename
    )


# len(df1_labelled)
# Loop through labeled data
for i in tqdm(range(len(df1_labelled))):
    
    try:
        stations_id = df1_labelled['stations'].values[i]
        location = "*"
        start_time = obspy.UTCDateTime(df1_labelled['timestamp'].values[i]) - 100
        end_time = start_time + 300

        
        stn_probs_seismiccnn_1d, big_reshaped_data, big_station_ids = process_model(
            model_seismiccnn_1d, stations_id, location, start_time, end_time, 
            one_d= True, model_type='dl', filename='P_10_30_F_05_15_50'
        )
        
        stn_probs_seismiccnn_2d, _, _ = process_model(
            model_seismiccnn_2d, stations_id, location, start_time, end_time, 
            one_d=False, model_type='dl', filename='P_10_30_F_05_15_50'
        )
        
        stn_probs_quakexnet_1d, _, _ = process_model(
            model_quakexnet_1d, stations_id, location, start_time, end_time, 
            one_d= True, model_type='dl', filename='P_10_30_F_05_15_50'
        )
        
        stn_probs_quakexnet_2d, _, _ = process_model(
            model_quakexnet_2d, stations_id, location, start_time, end_time, 
            one_d=False, model_type='dl', filename='P_10_30_F_05_15_50'
        )
        
        
        
        
        probs_seismiccnn_1d.append(stn_probs_seismiccnn_1d)
        probs_seismiccnn_2d.append(stn_probs_seismiccnn_2d)
        probs_quakexnet_1d.append(stn_probs_quakexnet_1d)
        probs_quakexnet_2d.append(stn_probs_quakexnet_2d)
        
        
        model = []
        # Machine Learning model probabilities
        stn_probs_ml_40, _, _ = process_model(
            model, stations_id, location, start_time, end_time, 
            one_d=False, model_type='ml', filename='P_10_30_F_05_15_50'
        )
        stn_probs_ml_110, _, _ = process_model(
            model, stations_id, location, start_time, end_time, 
            one_d=False, model_type='ml', filename='P_10_100_F_05_15_50'
        )
        stn_probs_ml_150, _, _ = process_model(
            model, stations_id, location, start_time, end_time, 
            one_d=False, model_type='ml', filename='P_50_100_F_05_15_50'
        )

        # Store machine learning probabilities
        probs_ml_40.append(stn_probs_ml_40)
        probs_ml_110.append(stn_probs_ml_110)
        probs_ml_150.append(stn_probs_ml_150)

        # Store reshaped data and station IDs
        station_data.append(big_reshaped_data)
        station_ids.append(big_station_ids)
        
        
        evids.append(df1_labelled['event_id'].values[i])
    except:
        pass

 


# Save lists to disk
with open('../results/alex_real_time/probs_seismiccnn_1d.pkl', 'wb') as f:
    pickle.dump(probs_seismiccnn_1d, f)

with open('../results/alex_real_time/probs_seismiccnn_2d.pkl', 'wb') as f:
    pickle.dump(probs_seismiccnn_2d, f)
    
with open('../results/alex_real_time/probs_quakexnet_1d.pkl', 'wb') as f:
    pickle.dump(probs_quakexnet_1d, f)
    
with open('../results/alex_real_time/probs_quakexnet_2d.pkl', 'wb') as f:
    pickle.dump(probs_quakexnet_2d, f)
    
with open('../results/alex_real_time/probs_ml_40.pkl', 'wb') as f:
    pickle.dump(probs_ml_40, f)
    
with open('../results/alex_real_time/probs_ml_110.pkl', 'wb') as f:
    pickle.dump(probs_ml_110, f)
    
with open('../results/alex_real_time/probs_ml_150.pkl', 'wb') as f:
    pickle.dump(probs_ml_150, f)
    
with open('../results/alex_real_time/station_data.pkl', 'wb') as f:
    pickle.dump(station_data, f)
with open('../results/alex_real_time/station_ids.pkl', 'wb') as f:
    pickle.dump(station_ids, f)
with open('../results/alex_real_time/evids.pkl', 'wb') as f:
    pickle.dump(evids, f)

print("Data saved successfully.")




## removing the downloaded files to clean up the directory. 

# Get the current working directory
directory = os.getcwd()

# List the files in the directory
files = os.listdir(directory)

# Iterate over the files and delete CSV files
for file in files:
    if file.endswith(".csv"):
        file_path = os.path.join(directory, file)
        os.remove(file_path)

        
for file in files:
    if file.endswith(".joblib"):
        file_path = os.path.join(directory, file)
        os.remove(file_path)