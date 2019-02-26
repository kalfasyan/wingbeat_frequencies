#!/usr/bin/env python
# coding: utf-8

import os, math
import numpy as np
seed = 2018
np.random.seed(seed)

import librosa
from scipy import signal

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers.merge import Concatenate
from keras.layers import Activation
from keras.layers import Dense

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger

from keras import Model
from keras import backend as K

from keras.utils import np_utils
#from keras.preprocessing import image
from keras.preprocessing.image import random_shift

from keras.applications.densenet import DenseNet121

from wavhandler import *
from utils import *

current_model = DenseNet121

model_name = 'wingbeats_manyinputs_' + current_model.__name__

best_weights_path = TEMP_DATADIR + model_name + '.h5'
log_path = TEMP_DATADIR + model_name + '.log'
monitor = 'val_acc'
batch_size = 32
epochs = 100
es_patience = 7
rlr_patience = 3

SR = 8000
N_FFT = 256
HOP_LEN = int(N_FFT / 6)
input_shape = (129, 120, 1)
target_names = mosquitos_6

# DATADIR = '/home/kalfasyan/data/insects/Wingbeats/'
DATADIR = '/data/leuven/314/vsc31431/insects/Wingbeats/'
X_names, y = get_data(filedir=DATADIR, target_names=target_names, only_names=True)

X_names, y = shuffle(X_names, y, random_state = seed)
X_train, X_test, y_train, y_test = train_test_split(X_names, y, stratify = y, test_size = 0.20, random_state = seed)

print ('train #recs = ', len(X_train))
print ('test #recs = ', len(X_test))

names = pd.DataFrame(X_train, columns=['name'])
df_train = pd.DataFrame(names)

df_train['filename'] = df_train['name'].str.extract('([F]\w{0,})',expand=True)
df_train['hour'] = df_train.filename.str.extract('([_]\w{0,2})',expand=True)
df_train['hour'] = df_train.hour.str.split('_',expand=True)[1].astype(int)
df_train_list = df_train.hour.tolist()

names = pd.DataFrame(X_test, columns=['name'])
df_test = pd.DataFrame(names)

df_test['filename'] = df_test['name'].str.extract('([F]\w{0,})',expand=True)
df_test['hour'] = df_test.filename.str.extract('([_]\w{0,2})',expand=True)
df_test['hour'] = df_test.hour.str.split('_',expand=True)[1].astype(int)
df_test_list = df_test.hour.tolist()

# def random_data_shift(data, u = 0.5):
#     if np.random.random() < u:
#         data = np.roll(data, int(round(np.random.uniform(-(len(data)), (len(data))))))
#     return data

def train_generator():
    while True:
        for start in range(0, len(X_train), batch_size):
            x_batch = []
            y_batch = []
            x_df_batch = []

            end = min(start + batch_size, len(X_train))
            train_batch = X_train[start:end]
            labels_batch = y_train[start:end]
            train_df_batch = df_train_list[start:end]

            for i in range(len(train_batch)):
                data, rate = librosa.load(train_batch[i], sr = SR)

                #data = random_data_shift(data, u = 1.0)

                data = librosa.stft(data, n_fft = N_FFT, hop_length = HOP_LEN)
                data = librosa.amplitude_to_db(data)

                data = np.flipud(data)

                data = np.expand_dims(data, axis = -1)
                data = random_shift(data, 0.25, 0.0, row_axis = 0, col_axis = 1, channel_axis = 2, fill_mode = 'constant', cval = np.min(data))

                # data = np.squeeze(data, axis = -1)
                # plt.imshow(data, cmap = 'gray')
                # plt.show()
                # data = np.expand_dims(data, axis = -1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])
                x_df_batch.append(train_df_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            x_df_batch = np.array(x_df_batch, np.float32)

            y_batch = np_utils.to_categorical(y_batch, len(target_names))
            x_df_batch = np_utils.to_categorical(x_df_batch, 24)

            yield [x_batch, x_df_batch], y_batch

def valid_generator():
    while True:
        for start in range(0, len(X_test), batch_size):
            x_batch = []
            y_batch = []
            x_df_batch = []

            end = min(start + batch_size, len(X_test))
            test_batch = X_test[start:end]
            labels_batch = y_test[start:end]
            test_df_batch = df_test_list[start:end]

            for i in range(len(test_batch)):
                data, rate = librosa.load(test_batch[i], sr = SR)

                data = librosa.stft(data, n_fft = N_FFT, hop_length = HOP_LEN)
                data = librosa.amplitude_to_db(data)

                data = np.flipud(data)

                data = np.expand_dims(data, axis = -1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])
                x_df_batch.append(test_df_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            x_df_batch = np.array(x_df_batch, np.float32)

            y_batch = np_utils.to_categorical(y_batch, len(target_names))
            x_df_batch = np_utils.to_categorical(x_df_batch, 24)

            yield [x_batch, x_df_batch], y_batch

img_input = Input(shape = input_shape)

model = current_model(include_top = True, weights = None, input_tensor = img_input)

x = model.get_layer(model.layers[-2].name).output#model.output

meta_input = Input(shape = [24])
y = BatchNormalization() (meta_input)

xy = (Concatenate()([x, y]))

xy = Dense(len(target_names)) (xy)
outputs = Activation('softmax') (xy)

model = Model([img_input, meta_input], [outputs])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

callbacks_list = [ModelCheckpoint(monitor = monitor,
                                filepath = best_weights_path,
                                save_best_only = True,
                                save_weights_only = True,
                                verbose = 1),
                    EarlyStopping(monitor = monitor,
                                patience = es_patience,
                                verbose = 1),
                    ReduceLROnPlateau(monitor = monitor,
                                factor = 0.1,
                                patience = rlr_patience,
                                verbose = 1),
                    CSVLogger(filename = log_path)]

model.fit_generator(train_generator(),
    steps_per_epoch = int(math.ceil(float(len(X_train)) / float(batch_size))),
    validation_data = valid_generator(),
    validation_steps = int(math.ceil(float(len(X_test)) / float(batch_size))),
    epochs = epochs,
    callbacks = callbacks_list,
    shuffle = False)

model.load_weights(best_weights_path)

loss, acc = model.evaluate_generator(valid_generator(),
        steps = int(math.ceil(float(len(X_test)) / float(batch_size))))

#print('loss:', loss)
print('Test accuracy:', acc)
