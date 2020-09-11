#!/usr/bin/env python
# coding: utf-8

# In[2]:


# get_ipython().run_line_magic('reset', '-f')
import sys
sys.path.insert(0, "..")
from wavhandler import Dataset
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report, make_scorer, log_loss
from utils_train import *
import deepdish as dd
from configs import DatasetConfiguration
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed=seed)

splitting = 'random'
data_setting = 'rawflt'
model_setting = 'conv1d'

# assert splitting in ['random','randomcv','custom'], "Wrong splitting method given."
# assert data_setting in ['raw','stft','psd_dB', 'cwt'], "Wrong data settting given."
# assert model_setting in ['wavenet','lstm','gru','conv1d','conv1d_psd',
#                         'DenseNet121','DenseNet169','DenseNet201',
#                         'InceptionResNetV2','VGG16','VGG19',
#                         'dl4tsc_fcn','dl4tsc_res', 'tsc_res_baseline',
#                         'tsc_fcn_baseline', 'conv1d_baseline', 'dl4tsc_inc'], "Wrong model setting given"


# In[3]:


data = Dataset('Wingbeats')
print(data.target_classes)

print(f'SPLITTING DATA {splitting}')
X_train, X_val, X_test, y_train, y_val, y_test, le = mosquito_data_split(splitting=splitting, dataset=data, downsampling=True, return_label_encoder=True)


# In[4]:


dataset = data
flag = ''
traincf = TrainConfiguration(nb_classes=6, setting=data_setting, model_name=f'MosquitoNET_{data_setting}_{model_setting}_{splitting}_{flag}')
using_conv2d = False

model = ModelConfiguration(model_setting=model_setting, data_setting=data_setting, nb_classes=6).config

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Actual training
h = model.fit(train_generator(X_train, y_train, 
                                    batch_size=traincf.batch_size,
                                    target_names=np.unique(y_test).tolist(),
                                    setting=traincf.setting,
                                    preprocessing_train_stats='',
                                    using_conv2d=using_conv2d),
                    steps_per_epoch = int(math.ceil(float(len(X_train)) / float(traincf.batch_size))),
                    epochs = traincf.epochs,
                    validation_data = valid_generator(X_val, y_val,
                                                        batch_size=traincf.batch_size,
                                                        target_names=np.unique(y_test).tolist(),
                                                        setting=traincf.setting,
                                                        preprocessing_train_stats='',
                                                        using_conv2d=using_conv2d),
                    validation_steps=int(math.ceil(float(len(X_test))/float(traincf.batch_size))),
                    callbacks=traincf.callbacks_list,
                    use_multiprocessing=False,
                    workers=1,
                    max_queue_size=32)


# In[ ]:


train_loss = h.history['loss']
train_score = h.history['accuracy']
val_loss = h.history['val_loss']
val_score = h.history['val_accuracy']
lr = h.history['lr']

# LOADING TRAINED WEIGHTS
model.load_weights(traincf.top_weights_path)

y_pred = model.predict(valid_generator(X_test, 
                                                y_test, 
                                                batch_size=traincf.batch_size, 
                                                setting=traincf.setting, 
                                                target_names=np.unique(y_test).tolist(),
                                                preprocessing_train_stats='',
                                                using_conv2d=using_conv2d),
        steps = int(math.ceil(float(len(X_test)) / float(traincf.batch_size))))


# In[ ]:




