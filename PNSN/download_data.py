#!/home/ahutko/miniconda3/envs/surface_dl/bin/python

#----- Download .joblib models and scaler_params*.csv files. Only do this once.
doi = '10.5281/zenodo.13334838'
from zenodo_get import zenodo_get
files = zenodo_get([doi])

