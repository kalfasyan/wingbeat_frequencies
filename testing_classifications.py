from utils import *
from wavhandler import *
import os
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import spearmanr
from scipy.spatial import distance
from utils_train import *
import argparse
import logging
import time

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--model_name", required=False,
    help="rgb vals of first image")
ap.add_argument("-s", "--setting", required=False,
    help="rgb vals of second image")    
args = vars(ap.parse_args())

print("Using {} model.".format(args['model_name']))
print("Using {} setting.".format(args['setting']))
time.sleep(1.)

dataset = Dataset('LG')
dataset.load(only_names=False, text_labels=False, nr_signals=np.inf)

XX = transform_data(dataset.X)#, setting='melbank')
print(XX.shape)
make_classification_ml(XX, dataset.y, clf_name=args["model_name"])

# make_classification_conv1d( dataset.X, 
#                             dataset.y, 
#                             model_name=args["model_name")

# make_classification_conv2d(dataset.filenames, 
#                             dataset.y, 
#                             model_name=args["model_name"], 
#                             setting=args["setting"])
