import pandas as pd
import numpy as np
from glob import glob
import obspy
from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt
from tqdm import tqdm

from obspy import UTCDateTime
from dateutil import parser
from pytz import timezone
import obspy
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader




# importing the dependencies. 

import scipy as sc
from scipy import signal
import h5py

from obspy.signal.filter import envelope

import tsfel
import random
from datetime import timedelta
import calendar
from tsfel import time_series_features_extractor
from sklearn.preprocessing import StandardScaler

from scipy import stats

#%config InlineBackend.figure_format = "png"

#from Feature_Extraction import compute_hibert

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# displaying all columns from pandas dataframe
# Set display options to show all columns
pd.set_option('display.max_columns', None)

from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load


import time
from datetime import datetime
import seaborn as sns

import sys
sys.path.append('../Common_Scripts/')

import seis_feature


sys.path.append('../Common_Scripts/')
from common_processing_functions import apply_cosine_taper
from common_processing_functions import butterworth_filter

import json
import os
from zenodo_get import zenodo_get
import yaml



# Load the configuration file
with open('../Common_Scripts/config.yml', 'r') as stream:
    config = yaml.safe_load(stream)


## Stations_id
stations_id = config['stations_id']
    
    
# Access the settings
common_dataset = pd.read_csv(config['common_dataset_path'])
scaler_params = pd.read_csv(config['scaler_params_path'])

lowcut = config['bandpass_filter']['lowcut']
highcut = config['bandpass_filter']['highcut']
fs = config['bandpass_filter']['fs']
num_corners = config['bandpass_filter']['num_corners']


## Number of CPUs to use. 
num_cpus = config['num_cpus']




# starttime from where we want to run the detector
starttime = obspy.UTCDateTime(config['starttime'])





# duration (in seconds) for which we want to run our detector
dur = config['dur']


# sampling frequency
samp_freq = config['samp_freq']

# input window length 
win = config['win']

# stride for the detector (time x freq)
stride = config['stride']


# Defining the client
client = Client(config['iris_client'])


# Defining the location code 
location = config['location']




# Continue accessing other settings as needed...




## Loading the configuration file that contains features we want to extract. 

# Specify the file path where you want to save the JSON file
file_path = '../Common_Scripts/cfg_file_reduced.json'

# Load the JSON data from the file
with open(file_path, 'r') as f:
    cfg_file_sample = json.load(f)



    
## Downloading the trained model from zenodo

doi = '10.5281/zenodo.11043908'  #Downloading the model trained on top 50 features
files = zenodo_get([doi])




# Later, you can load the model from disk
loaded_model = load('best_rf_model_top_50_features_50_100.joblib')


## Deleting the downloaded file since its very large
# Specify the file path
file_path = 'best_rf_model_top_50_features_50_100.joblib'

# Check if the file exists
if os.path.exists(file_path):
    # Remove the file
    os.remove(file_path)
    print(f"File '{file_path}' removed successfully.")
else:
    print(f"File '{file_path}' does not exist.")

    
    
def surface_event_detection():

    # grabbing the columns of common dataset. 
    columns = common_dataset.columns[1:]
    st_data_full = []
    result_stns = []
    index_stns = []
    prob_stns = []

    st_overall = []
    st_overall_data = []
    st_overall_times = []



    for stn_id in tqdm(stations_id):


        ## Extracting the station
        stn = stn_id.split('.')[1]

        ## Extracting the network
        network = stn_id.split('.')[0]



        st = []




        try:
            # Attempt to get waveform using 'EHZ' channel
            st = client.get_waveforms(starttime=starttime, endtime=starttime + dur, station=stn,
                                      network=network, channel='EHZ', location=location)
        except:
            try:
                # Attempt to get waveform using 'BHZ' channel
                st = client.get_waveforms(starttime=starttime, endtime=starttime + dur, station=stn,
                                          network=network, channel='BHZ', location=location)
            except:
                try:
                    # Use 'HHZ' channel as a fallback option
                    st = client.get_waveforms(starttime=starttime, endtime=starttime + dur, station=stn,
                                              network=network, channel='HHZ', location=location)
                except:
                    pass


        try:

            # resampling all the data to 100 Hz since thats 
            st = st.resample(samp_freq) 

            st.detrend()

            st_data_full = []

            ## Ideally the data should come in single stream (len(st) == 1) but if it comes in multiple streams
            ## We will take the first stream. 
            times = st[0].times()
            for i in range(len(st)):
                st_data_full = np.hstack([st_data_full, st[i].data])


                if i+1 < len(st):
                    diff = st[i+1].stats.starttime - st[i].stats.endtime

                    times = np.hstack((times, st[i+1].times()+times[-1]+diff))




                    #print('Final times')


            st_overall_data.append(st_data_full)
            st_overall.append(st)
            st_overall_times.append(times)
            trace_data = [st_data_full[i:i+int(win*samp_freq)] for i in tqdm(range(0, len(st_data_full), stride)) if len(st_data_full[i:i+int(win*samp_freq)]) == int(win*samp_freq)]
            trace_times = [times[i] for i in tqdm(range(0, len(st_data_full), stride)) if len(st_data_full[i:i+int(win*samp_freq)]) == int(win*samp_freq)]



            trace_data = np.array(trace_data)
            tapered = apply_cosine_taper(trace_data)
            filtered = butterworth_filter(tapered, lowcut, highcut, fs, num_corners, filter_type='bandpass')

            # Applying the normalization.
            norm = filtered / np.max(abs(np.stack(filtered)), axis=1)[:, np.newaxis]

            result = []
            prob = []
            time = []
            index = []

            for i in tqdm(range(len(norm))):


                tsfel_features = time_series_features_extractor(cfg_file_sample, norm[i], fs=100)
                tr_full = obspy.Trace(norm[i])
                tr_full.stats.sampling_rate = 100
                physical_features = seis_feature.compute_physical_features(tr=tr_full, envfilter=False)

                final_features = pd.concat([tsfel_features, physical_features], axis=1)
                final_features['hod'] = (starttime).hour - 8
                final_features['dow'] = (starttime).weekday
                final_features['moy'] = (starttime).month

                features = final_features.loc[:, columns]
                scaler_params.index = scaler_params.iloc[:,1]
                final_scaler_params = scaler_params.loc[features.columns]

                for k in range(len(features.columns)):
                    features.iloc[:, k] = (features.iloc[:, k] - final_scaler_params.iloc[k, 2]) / final_scaler_params.iloc[k, 3]

                # Check for NaN values in features
                if features.isnull().values.any():
                    print(f"NaN values detected in iteration {i}. Skipping prediction.")

                #features['E_20_50'] = 0.001
                # extracting the results.
                result.append(loaded_model.predict(features))
                prob.append(loaded_model.predict_proba(features))
                index.append(i)


            result_stns.append(result)
            index_stns.append(index)
            prob_stns.append(prob)
        except:
            pass
        
    return result_stns, index_stns, prob_stns, st_overall, st_overall_data, st_overall_times



