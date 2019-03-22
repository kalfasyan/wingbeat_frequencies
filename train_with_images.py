
import glob, os, sys, io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image
import math

import keras
from keras.layers import Input
from keras.applications.densenet import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.utils import np_utils

from wavhandler import *
from utils import *

import logging
logger = logging.getLogger()
logger.propagate = False
logger.setLevel(logging.ERROR)
np.random.seed(0)
import seaborn as sns
sns.set()

seed = 2018
np.random.seed(seed)



current_model = DenseNet121

model_name = TEMP_DATADIR + 'wingbeats_imgs' + current_model.__name__

best_weights_path = model_name + '.h5'
log_path = model_name + '.log'
monitor = 'val_acc'
batch_size = 32
epochs = 100
es_patience = 7
rlr_patience = 3

SR = 8000
N_FFT = 256
HOP_LEN = int(N_FFT / 6)
input_shape = (129, 120, 1)



X_fnames, y = get_data(dataset='MOSQUITOES_IMGS_train', nr_signals=np.inf, only_names=True, text_labels=False)

X_train, y_train = shuffle(X_fnames, y, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y, stratify = y, test_size = 0.20, random_state = seed)



def shift_roll(data, u, shift_pct=0.006, axis=0):
    if np.random.random() < u:
        data = np.roll(data, int(round(np.random.uniform(-(len(data)*shift_pct), (len(data)*shift_pct)))), axis=axis)
    return data

def train_generator(X=X_train, y=y_train):
    while True:
        for start in range(0, len(X), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(X))
            train_batch = X[start:end]
            labels_batch = y[start:end]

            for i in range(len(train_batch)):
                temp = Image.open(train_batch[i])
                data = np.array(temp.copy())[:,:,0]
                temp.close()
                data = np.expand_dims(data, axis=-1)
                
                data = shift_roll(data, u=0.5, shift_pct=0.006, axis=0)
                data = shift_roll(data, u=0.5, shift_pct=0.25, axis=1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)

            y_batch = np_utils.to_categorical(y_batch, len(np.unique(y)))

            yield x_batch, y_batch

def valid_generator(X=X_val, y=y_val):
    while True:
        for start in range(0, len(X), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(X))
            val_batch = X[start:end]
            labels_batch = y[start:end]

            for i in range(len(val_batch)):
                temp = Image.open(val_batch[i])
                data = np.array(temp.copy())[:,:,0]
                temp.close()
                data = np.expand_dims(data, axis=-1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)

            y_batch = np_utils.to_categorical(y_batch, len(np.unique(y)))

            yield x_batch, y_batch


img_input = Input(shape = input_shape)

model = current_model(input_tensor = img_input, classes = len(DataSet('MOSQUITOES').names), weights = None)

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



model.fit_generator(
    train_generator(),
    steps_per_epoch = int(math.ceil(float(len(X_train)) / float(batch_size))),
    validation_data = valid_generator(), 
    validation_steps = int(math.ceil(float(len(X_val)) / float(batch_size))),
    epochs = 100,
    callbacks=callbacks_list,
    use_multiprocessing=True,
    shuffle=True)



# import math
# model.load_weights(best_weights_path)

# loss, acc = model.evaluate_generator(valid_generator,
#         steps = int(math.ceil(float(len(X_test)) / float(batch_size))))

# #print('loss:', loss)
# print('Test accuracy:', acc)



# filenames, y = get_data(dataset='MOSQUITOES_IMGS_train', nr_signals=np.inf, only_names=True)

# X_names, y = shuffle(filenames, y, random_state = seed)

# X_tmp, _, y_tmp = get_data(dataset='MOSQUITOES_IMGS_train', nr_signals=10, only_names=False, text_labels=True)

# train_data_dir = os.path.join(BASE_DIR, 'Wingbeats_spectrograms/train/')

# datagen = ImageDataGenerator(rescale=1./255,
#                             width_shift_range=0.2,
#                             height_shift_range=0.05,
#                             validation_split=0.2)

# # train_generator = datagen.flow(X_train, y_train, batch_size=32, shuffle=True, subset='training')
# # valid_generator = datagen.flow(X_train, y_train, batch_size=32, shuffle=True, subset='validation'))

# train_generator = datagen.flow_from_directory(train_data_dir,
#                                             target_size=(129,120),
#                                             batch_size=32,
#                                             shuffle=True,
#                                             subset='training')

# valid_generator = datagen.flow_from_directory(train_data_dir,
#                                               target_size=(129,120),
#                                               batch_size=32,
#                                               subset='validation')

# # fit parameters from data
# datagen.fit(X_tmp)

# import matplotlib.pyplot as plt
# # configure batch size and retrieve one batch of images
# for X_batch, y_batch in datagen.flow(np.stack(X_tmp, axis=0), y_tmp, batch_size=11):
#     # create a grid of 3x3 images
#     plt.figure(figsize=(20,12))
#     plt.grid(False)
#     for i in range(0, 9):
#         plt.subplot(340 + 1 + i)
#         plt.grid(False)
#         plt.imshow(X_batch[i].reshape(129, 120,3), cmap=plt.get_cmap('gray'))
#     # show the plot
#     plt.show()
#     break

# from keras.layers import Input
# from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger


