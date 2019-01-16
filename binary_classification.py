#!/usr/bin/env python
# coding: utf-8

import glob, os, sys, io
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from sklearn.model_selection import train_test_split
import keras

from wavhandler import *
from utils import *

import logging
logger = logging.getLogger()
logger.propagate = False
logger.setLevel(logging.ERROR)
np.random.seed(0)
import seaborn as sns
sns.set()

import librosa
from scipy import signal
import math
seed = 2018
np.random.seed(seed)
from matplotlib import pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.layers import Input
from keras import layers

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger

from keras import Model
from keras import backend as K

from keras.utils import np_utils
from keras.preprocessing import image

from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception


# In[2]:


target_names = all_6
current_model = DenseNet121
model_name = './data/wingbeats_' + current_model.__name__

best_weights_path = model_name + '.h5'
log_path = model_name + '.log'
monitor = 'val_acc'
batch_size = 16
epochs = 100
es_patience = 7
rlr_patience = 2

SR = 8000
N_FFT = 256
HOP_LEN = int(N_FFT / 6)
input_shape = (129, 120, 1)

img_input = Input(shape = input_shape)
pretrained = current_model(input_tensor = img_input,
                           classes = len(target_names),
                           weights = best_weights_path)

bi_classes = ['LG_drosophila_10_09', 'LG_zapr_26_09']
target_names = bi_classes
X_names,y = get_data(target_names=bi_classes, nr_signals=999999, only_names=True)

X_names, y = shuffle(X_names, y, random_state = seed)
X_train, X_test, y_train, y_test = train_test_split(X_names, y, stratify = y, test_size = 0.20, random_state = seed)

out = pretrained.get_layer(pretrained.layers[-2].name).output
out = layers.Dense(128, activation='relu')(out)
out = layers.Dropout(0.5)(out)
out = layers.Dense(2, activation='softmax')(out)
new_model = Model(inputs=img_input, outputs=out)

for layer in new_model.layers[:-2]:
    layer.trainable = False
for layer in new_model.layers[-10:]:
    print(layer, layer.trainable)

new_model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

def shift(x, wshift, hshift, row_axis = 0, col_axis = 1, channel_axis = 2, fill_mode = 'constant', cval = 0.):
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = hshift * h
    ty = wshift * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    transform_matrix = translation_matrix
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def random_data_shift(data, w_limit = (-0.25, 0.25), h_limit = (-0.0, 0.0), cval = 0., u = 0.5):
    if np.random.random() < u:
        wshift = np.random.uniform(w_limit[0], w_limit[1])
        hshift = np.random.uniform(h_limit[0], h_limit[1])
        data = shift(data, wshift, hshift, cval = cval)
    return data

def train_generator():
    while True:
        for start in range(0, len(X_train), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(X_train))
            train_batch = X_train[start:end]
            labels_batch = y_train[start:end]

            for i in range(len(train_batch)):
                data, rate = librosa.load(train_batch[i], sr = SR)

                #data = random_data_shift(data, u = 1.0)

                data = librosa.stft(data, n_fft = N_FFT, hop_length = HOP_LEN)
#                 data = librosa.amplitude_to_db(data)
                data = np.abs(data)
                data = np.flipud(data)

                data = np.expand_dims(data, axis = -1)
                data = random_data_shift(data, w_limit = (-0.25, 0.25), h_limit = (-0.0, 0.0), cval = np.min(data), u = 1.0)

                # data = np.squeeze(data, axis = -1)
                # plt.imshow(data, cmap = 'gray')
                # plt.show()
                # data = np.expand_dims(data, axis = -1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)

            y_batch = np_utils.to_categorical(y_batch, len(target_names))

            yield x_batch, y_batch

def valid_generator():
    while True:
        for start in range(0, len(X_test), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(X_test))
            test_batch = X_test[start:end]
            labels_batch = y_test[start:end]

            for i in range(len(test_batch)):
                data, rate = librosa.load(test_batch[i], sr = SR)

                data = librosa.stft(data, n_fft = N_FFT, hop_length = HOP_LEN)
#                 data = librosa.amplitude_to_db(data)
                data = np.abs(data)
                data = np.flipud(data)

                data = np.expand_dims(data, axis = -1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)

            y_batch = np_utils.to_categorical(y_batch, len(target_names))

            yield x_batch, y_batch

callbacks_list = [ModelCheckpoint(monitor = monitor,
                                filepath = best_weights_path[:-3]+'_binary.h5',
                                save_best_only = True,
                                save_weights_only = True,
                                verbose = 1),
                    EarlyStopping(monitor = monitor,
                                patience = es_patience,
                                verbose = 1),
                    ReduceLROnPlateau(monitor = monitor,
                                factor = 0.2,
                                patience = rlr_patience,
                                verbose = 1),
                    CSVLogger(filename = log_path)]


# In[ ]:


new_model.fit_generator(train_generator(),
    steps_per_epoch = int(math.ceil(float(len(X_train)) / float(batch_size))),
    validation_data = valid_generator(),
    validation_steps = int(math.ceil(float(len(X_test)) / float(batch_size))),
    epochs = epochs,
    callbacks = callbacks_list,
    shuffle = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
