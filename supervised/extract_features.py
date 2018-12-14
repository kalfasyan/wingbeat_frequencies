import glob, os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
from wavhandler import *
from natsort import natsorted
from utils import *
import matplotlib.pyplot as plt
import numpy as np

smpl = -1

aedes = WavHandler('/home/yannis/data/insects/Potamitis/Wingbeats/Aedes', sample_size=smpl, recursive=True)
anoph = WavHandler('/home/yannis/data/insects/Potamitis/Wingbeats/Anopheles', sample_size=smpl, recursive=True)
culex = WavHandler('/home/yannis/data/insects/Potamitis/Wingbeats/Culex', sample_size=smpl, recursive=True)

aedes.read(); aedes.preprocess(); 
anoph.read(); anoph.preprocess(); 
culex.read(); culex.preprocess(); 

final_aedes = tsfresh_transform(aedes.df_signals); print('Aedes data ready..')
final_anoph = tsfresh_transform(anoph.df_signals); print('Anoph data ready..')
final_culex = tsfresh_transform(culex.df_signals); print('Culex data ready..')

# extracting features using tsfresh
from tsfresh import extract_features
#settings = make_tsfresh_settings()
features_aedes = extract_features(final_aedes, column_id='id', column_sort='time')#, kind_to_fc_parameters = settings)
features_aedes.set_index(aedes.df_signals.columns, inplace=True)
features_anoph = extract_features(final_anoph, column_id='id', column_sort='time')#, kind_to_fc_parameters = settings)
features_anoph.set_index(anoph.df_signals.columns, inplace=True)
features_culex = extract_features(final_culex, column_id='id', column_sort='time')#, kind_to_fc_parameters = settings)
features_culex.set_index(culex.df_signals.columns, inplace=True)

#from tsfresh.utilities.dataframe_functions import impute
#impute(extracted_features)

featurelist = [features_aedes, 
				features_anoph, 
				features_culex]

df = pd.concat(featurelist, axis=0)
df.to_pickle('./data/combined_all.pkl'.format(smpl))

# WHEN YOU HAVE LABELS
#from tsfresh import extract_relevant_features
#features_filtered_direct = extract_relevant_features(timeseries, y, column_id='id', column_sort='time')

#data_100 = pd.concat([features_zapr, features_dros], axis=0)