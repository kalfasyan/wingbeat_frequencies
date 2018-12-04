import glob, os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
from wavhandler import *
from natsort import natsorted
from utils import *
import matplotlib.pyplot as plt
import numpy as np

smpl = 20000

aedes = WavHandler('/home/yannis/data/insects/Potamitis/Wingbeats/Aedes', sample_size=smpl, recursive=True)
anoph = WavHandler('/home/yannis/data/insects/Potamitis/Wingbeats/Anopheles', sample_size=smpl, recursive=True)
culex = WavHandler('/home/yannis/data/insects/Potamitis/Wingbeats/Culex', sample_size=smpl, recursive=True)

aedes.read(); aedes.preprocess(); aedes.filter_accepted_signals(); aedes.crop() 
anoph.read(); anoph.preprocess(); anoph.filter_accepted_signals(); anoph.crop() 
culex.read(); culex.preprocess(); culex.filter_accepted_signals(); culex.crop()

final_aedes = tsfresh_transform(aedes.df_cropped); print('Aedes data ready..')
final_anoph = tsfresh_transform(anoph.df_cropped); print('Anoph data ready..')
final_culex = tsfresh_transform(culex.df_cropped); print('Culex data ready..')

# extracting features using tsfresh
from tsfresh import extract_features
#settings = make_tsfresh_settings()
features_aedes = extract_features(final_aedes, column_id='id', column_sort='time')#, kind_to_fc_parameters = settings)
features_anoph = extract_features(final_anoph, column_id='id', column_sort='time')#, kind_to_fc_parameters = settings)
features_culex = extract_features(final_culex, column_id='id', column_sort='time')#, kind_to_fc_parameters = settings)

#from tsfresh.utilities.dataframe_functions import impute
#impute(extracted_features)

features_aedes['label'] = 'aedes'
features_anoph['label'] = 'anoph'
features_culex['label'] = 'culex'

featurelist = [features_aedes, 
				features_anoph, 
				features_culex]

df = pd.concat(featurelist, axis=0)
df.to_pickle('combined_{}.pkl'.format(smpl))

# WHEN YOU HAVE LABELS
#from tsfresh import extract_relevant_features
#features_filtered_direct = extract_relevant_features(timeseries, y, column_id='id', column_sort='time')

#data_100 = pd.concat([features_zapr, features_dros], axis=0)