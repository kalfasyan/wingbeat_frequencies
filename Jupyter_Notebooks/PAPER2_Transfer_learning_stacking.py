#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from tensorflow.keras.models import Model

seed = 42
np.random.seed(seed=seed)

def train_generator(X_train, y_train, batch_size, target_names, setting='stft'):
    from tensorflow.keras import utils
    while True:
        for start in range(0, len(X_train), batch_size):
            x_batch = []
            y_batch = []            
            end = min(start + batch_size, len(X_train))
            train_batch = X_train[start:end]
            labels_batch = y_train[start:end]
            for i in range(len(train_batch)):
                data, _ = librosa.load(train_batch[i], sr = SR)
                data = metamorphose(data, setting=setting)
                data = np.expand_dims(data, axis = -1)
                if using_conv2d:
                    data = np.expand_dims(data, axis = -1)
                x_batch.append(data)
                y_batch.append(labels_batch[i])
            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            y_batch = utils.to_categorical(y_batch, len(target_names))
            yield x_batch, y_batch

def valid_generator(X_val, y_val, batch_size, target_names, setting='stft'):
    from tensorflow.keras import utils
    while True:
        for start in range(0, len(X_val), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(X_val))
            test_batch = X_val[start:end]
            labels_batch = y_val[start:end]
            for i in range(len(test_batch)):
                data, _ = librosa.load(test_batch[i], sr = SR)
                data = metamorphose(data, setting=setting)
                # Expand dimensions
                data = np.expand_dims(data, axis = -1)
                if using_conv2d:
                    data = np.expand_dims(data, axis = -1)
                x_batch.append(data)
                y_batch.append(labels_batch[i])
            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            y_batch = utils.to_categorical(y_batch, len(target_names))
            yield x_batch, y_batch

splitting = 'custom'
data_setting = 'raw'
model_setting = 'conv1d'


# ### Splitting mosquito data same way it was split to train

# In[2]:


d = Dataset('Wingbeats')
d.read(loadmat=False)

X_train, X_val, X_test, y_train, y_val, y_test, le = mosquito_data_split(splitting=splitting, dataset=d, downsampling=False, return_label_encoder=True)

X_train, X_val, y_train, y_val = X_train[0], X_val[0], y_train[0], y_val[0]
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train, random_state=seed)
X_val, y_val = shuffle(X_val, y_val, random_state=seed)

get_labelencoder_mapping(le)


# ### Loading MosquitoNet and removing its last 2 layers

# In[3]:


merged = True

using_conv2d = False
if merged:
    # Merging the mosquito genuses together
    y_train = pd.Series(y_train).replace({1:0, 3:2, 5:4}).replace({2:1, 4:2}).tolist()
    y_val = pd.Series(y_val).replace({1:0, 3:2, 5:4}).replace({2:1, 4:2}).tolist()
    y_test = pd.Series(y_test).replace({1:0, 3:2, 5:4}).replace({2:1, 4:2}).tolist()

    # Defining model parameters
    modelname = f'TL_{splitting}_{data_setting}_{model_setting}_MERGED_weights'
    traincf = TrainConfiguration(dataset=d, setting=data_setting, model_name=modelname)
    d.target_classes = ['Aedes','Anopheles','Culex']
    traincf.target_names = np.unique(d.target_classes)
    traincf.targets = len(traincf.target_names)    
    model = ModelConfiguration(model_setting=model_setting, data_setting=data_setting, target_names=traincf.target_names).config
else:
    modelname = f'TL_{splitting}_{data_setting}_{model_setting}_weights'
    traincf = TrainConfiguration(dataset=d, setting=data_setting, model_name=modelname)
    model = ModelConfiguration(model_setting=model_setting, data_setting=data_setting, target_names=traincf.target_names).config

model.load_weights(TEMP_DATADIR+modelname+'.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# NOT LOADING MODELS
model.summary()

if data_setting in ['raw', 'rawflt','psd','psdflt', 'psd_dB','psd_dBflt']:
    # cut_mosquito_model is the model without its last dropout and softmax
    cut_mosquito_model = Model(model.inputs, model.layers[-3].output)
elif data_setting in ['stft','stftflt']:
    cut_mosquito_model = Model(model.inputs, model.layers[-2].output)


# In[4]:


cut_mosquito_model.summary()


# ## Reading Flies data and splitting into Train/Val/Test

# In[5]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
import seaborn as sb
from tensorflow.keras.layers import Input
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from sklearn.utils import class_weight
import deepdish as dd
from wavhandler import *


# In[6]:


dconf = Dataset('Pcfruit_sensor49')
dconf.read(loadmat=False);
dconf.get_sensor_features(temp_humd=False)
sub = dconf.df_features

mel_test_dates = ['20191216','20191217','20191220','20191221','20191222','20191223',
                 '20191224','20191225','20191226','20191227','20191228','20191229',
                 '20191230','20191231','20200101','20200102','20200103','20200104',
                 '20200105','20200106','20200107','20200108','20200109','20200110',
                 '20200111','20200112','20200113','20200114','20200115','20200116',
                 '20200117','20200118','20200119']
mel_val_dates = [['20200131','20200201'],['20200307'],['20200313','20200314','20200315','20200316']]
suz_test_dates = ['20200207', '20200208']
suz_val_dates = [['20200209'],['20200214'],['20200219']]


# In[7]:


sub['y'] = dconf.df_features.filenames.apply(lambda x: x.split('/')[dconf.class_path_idx])
df_suz = sub[sub['y'] == 'D. suzukii']
df_mel = sub[sub['y'] == 'D. melanogaster']

plt.figure(figsize=(20,4))
plt.subplot(121); df_mel.groupby('datestr')['filenames'].count().plot(kind="bar")
plt.subplot(122); df_suz.groupby('datestr')['filenames'].count().plot(kind="bar")

df_suz.head()


# In[8]:


mel_test = df_mel[df_mel.datestr.isin(mel_test_dates)][['filenames','y']]
suz_test = df_suz[df_suz.datestr.isin(suz_test_dates)][['filenames','y']]
# suz_test.sample(5)
# For the test set we made 1 dataframe for each insect species,
# but for the validation set we have a list of 3 dataframes for each species
# since we select three validation sets.
mel_val, suz_val = [],[]
for i in range(3):
    suz_val.append(df_suz[df_suz.datestr.isin(suz_val_dates[i])][['filenames','y']])
    mel_val.append(df_mel[df_mel.datestr.isin(mel_val_dates[i])][['filenames','y']])
# mel_val[0].sample(5)

lenc = LabelEncoder()
lenc.fit(dconf.y)

test = pd.concat([mel_test, suz_test])
Xf_test = test.filenames.tolist()
yf_test = lenc.transform(test.y).tolist()
# Trainval contains all data for training and validation sets [it excludes test set]
trainval = sub[~sub.filenames.isin(Xf_test)][['filenames','y']]
trainval.sample(5)


# # Function to create model, extract features and train

# In[9]:


def create_top_model(input_shape=cut_mosquito_model.output_shape[1:]):
    ### Creating a shallow model to put on top of Mosquito model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=input_shape))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(2, activation='softmax'))
    top_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return top_model

def extract_features_mosquitonet(Xf_train, yf_train, Xf_val, yf_val, setting=data_setting):
    ### Passing fly data through the cut Mosquito model to get output
    ##### This output will be used to train the shallow model for few epochs

    Xf_xtracted_train = cut_mosquito_model.predict(valid_generator(Xf_train, 
                                                    yf_train, 
                                                    batch_size=128, 
                                                    setting=data_setting, 
                                                    target_names=['suz','mel']),
                                                steps = int(math.ceil(float(len(Xf_train)) / float(128))))

    Xf_xtracted_val = cut_mosquito_model.predict(valid_generator(Xf_val, 
                                                    yf_val, 
                                                    batch_size=128, 
                                                    setting=data_setting, 
                                                    target_names=['suz','mel']),
                                                steps = int(math.ceil(float(len(Xf_val)) / float(128))))    
    return Xf_xtracted_train, Xf_xtracted_val

def train_model(Xf_train, yf_train, Xf_val, yf_val, Xf_xtracted_train, Xf_xtracted_val, 
                cut_mosquito_model, top_model, frozen_layers=0):
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(yf_train), y=yf_train)
    class_weights = {i : weights[i] for i in range(2)}
    
    ### Training Shallow model for a few epochs
    traincf_flies = TrainConfiguration(dataset=dconf, setting=data_setting, monitor='val_accuracy', 
                                       model_name=f'top_model_flies', batch_size=32, epochs=5)
    top_model.fit(Xf_xtracted_train, yf_train, 
                  validation_data=(Xf_xtracted_val,yf_val),
                  batch_size=traincf_flies.batch_size, 
                  epochs=traincf_flies.epochs, 
                  callbacks=traincf_flies.callbacks_list,
                 class_weight=class_weights,
                 verbose=1);
    ### Freezing first few layers of MosquitoNet
    if frozen_layers > 0:
        for lay in cut_mosquito_model.layers[:frozen_layers]:
            lay.trainable = False
    ### Adding shallow model on top of the cut Mosquito model
    inputA = Input(cut_mosquito_model.input_shape[1:])
    outputA = cut_mosquito_model(inputA)
    outputB = top_model(outputA)
    modelC = Model(inputA, outputB)
    modelC.compile(loss='mse',
                  optimizer='adam',#optimizers.SGD(lr=1e-3, momentum=0.9),
                  metrics=['accuracy'])
    ### Training MosquitoNet+Shallow
    traincf_flies = TrainConfiguration(dataset=dconf, setting=data_setting, monitor='val_accuracy', es_patience=4, rlr_patience=4, model_name=f'whole_model_flies', batch_size=32)
    h = modelC.fit(train_generator(Xf_train, yf_train, 
                                        batch_size=traincf_flies.batch_size,
                                        target_names=traincf_flies.target_names,
                                        setting=traincf_flies.setting),
                        steps_per_epoch = int(math.ceil(float(len(Xf_train)) / float(traincf_flies.batch_size))),
                        epochs = traincf_flies.epochs,
                        validation_data = valid_generator(Xf_val, yf_val,
                                                            batch_size=traincf_flies.batch_size,
                                                            target_names=traincf_flies.target_names,
                                                            setting=traincf_flies.setting),
                        validation_steps=int(math.ceil(float(len(Xf_test))/float(traincf_flies.batch_size))),
                        callbacks=traincf_flies.callbacks_list, 
                        class_weight=class_weights, 
                        verbose=1);

    yf_pred = modelC.predict(valid_generator(Xf_test, 
                            yf_test, 
                            batch_size=128, 
                            setting=data_setting, 
                            target_names=['suz','mel']),
                        steps = int(math.ceil(float(len(Xf_test)) / float(128))))
    return yf_pred, h, modelC


# In[10]:


## DEBUGGING
# valfold_mel = trainval[trainval.filenames.isin(mel_val[valset].filenames)]
# valfold_suz = trainval[trainval.filenames.isin(suz_val[valset].filenames)]
# validation = pd.concat([valfold_mel, valfold_suz])
# Xf_val = validation.filenames.tolist()
# yf_val = lenc.transform(validation.y)

# train = trainval[~trainval.filenames.isin(validation.filenames)]
# train_smpl = shuffle(train.sample(smpl, random_state=seed), random_state=seed)
# Xf_train, yf_train = train_smpl.filenames.tolist(), lenc.transform(train_smpl.y)

# next(valid_generator(Xf_train, 
#                     yf_train, 
#                     batch_size=128, 
#                     setting='raw', 
#                     target_names=['suz','mel']))[0].shape


# In[ ]:





# In[11]:


# smpl = 1000#[100, 175, 250, 325, 400, 500, 1000, 2000, 4000, 6000, 8000]
# frozen_layers = 19 #[0, 5, 15, 19]
# trial = 0 #[0,1,2,3,4,5,6,7,8,9]
# valset = 0

# savepath = f'../temp_data/{data_setting}_merged{merged}_nr{frozen_layers}_samples{smpl}_val{valset}_trial{trial}.csv'
# print(savepath)

# print(f'\n\n###########\nNOW USING {smpl} SAMPLES in {valset} VALIDATION FOLD\n##############')
# smpl_results = {}

# valfold_mel = trainval[trainval.filenames.isin(mel_val[valset].filenames)]
# valfold_suz = trainval[trainval.filenames.isin(suz_val[valset].filenames)]
# validation = pd.concat([valfold_mel, valfold_suz])
# Xf_val = validation.filenames.tolist()
# yf_val = lenc.transform(validation.y)

# train = trainval[~trainval.filenames.isin(validation.filenames)]
# train_smpl = shuffle(train.sample(smpl, random_state=seed), random_state=seed)
# Xf_train, yf_train = train_smpl.filenames.tolist(), lenc.transform(train_smpl.y)

# print(f"Train: \n{pd.Series(yf_train).value_counts()}")
# print(f"Val: \n{pd.Series(yf_val).value_counts()}")
# print(f"Test: \n{pd.Series(yf_test).value_counts()}")    

# # cut_mosquito_model = Model(model.inputs, model.layers[-3].output)
# top_model = create_top_model(input_shape=cut_mosquito_model.output_shape[1:])
# Xf_xtracted_train, Xf_xtracted_val = extract_features_mosquitonet(Xf_train, yf_train, Xf_val, yf_val)
# # yf_pred, h, modelC = train_model(Xf_train, yf_train, Xf_val, yf_val, Xf_xtracted_train, Xf_xtracted_val, 
# #                                  cut_mosquito_model, top_model, frozen_layers=frozen_layers)

# weights = class_weight.compute_class_weight('balanced', classes=np.unique(yf_train), y=yf_train)
# class_weights = {i : weights[i] for i in range(2)}

# ### Training Shallow model for a few epochs
# traincf_flies = TrainConfiguration(dataset=dconf, setting=data_setting, monitor='val_accuracy', 
#                                    model_name=f'top_model_flies', batch_size=32, epochs=10)

# top_model.fit(Xf_xtracted_train, yf_train, 
#               validation_data=(Xf_xtracted_val,yf_val),
#               batch_size=traincf_flies.batch_size, 
#               epochs=traincf_flies.epochs, 
# #               callbacks=traincf_flies.callbacks_list,
#              class_weight=class_weights,
#              verbose=1)

# ### Freezing first few layers of MosquitoNet
# if frozen_layers > 0:
#     for lay in cut_mosquito_model.layers[:frozen_layers]:
#         lay.trainable = False
# ### Adding shallow model on top of the cut Mosquito model
# inputA = Input(cut_mosquito_model.input_shape[1:])
# outputA = cut_mosquito_model(inputA)
# outputB = top_model(outputA)
# modelC = Model(inputA, outputB)
# modelC.compile(loss='binary_crossentropy',
#               optimizer='adam',#optimizers.SGD(lr=1e-3, momentum=0.9),
#               metrics=['accuracy'])
# ### Training MosquitoNet+Shallow
# traincf_flies = TrainConfiguration(dataset=dconf, setting=data_setting, monitor='val_accuracy', es_patience=4, rlr_patience=4, model_name=f'whole_model_flies', batch_size=32)

# h = modelC.fit(train_generator(Xf_train, yf_train, 
#                                     batch_size=traincf_flies.batch_size,
#                                     target_names=traincf_flies.target_names,
#                                     setting=traincf_flies.setting),
#                     steps_per_epoch = int(math.ceil(float(len(Xf_train)) / float(traincf_flies.batch_size))),
#                     epochs = traincf_flies.epochs,
#                     validation_data = valid_generator(Xf_val, yf_val,
#                                                         batch_size=traincf_flies.batch_size,
#                                                         target_names=traincf_flies.target_names,
#                                                         setting=traincf_flies.setting),
#                     validation_steps=int(math.ceil(float(len(Xf_test))/float(traincf_flies.batch_size))),
#                     callbacks=traincf_flies.callbacks_list, 
#                     class_weight=class_weights, 
#                     verbose=1)


# In[12]:


def myjob(trainval, mel_val, suz_val, Xf_test, yf_test, trial=0, frozen_layers=0, smpl=100, valset=0):
    savepath = f'../temp_data/{data_setting}_merged{merged}_nr{frozen_layers}_samples{smpl}_val{valset}_trial{trial}.csv'
    print(savepath)
    if os.path.isfile(savepath):
        print("ALREADY EXISTS")
        return

    print(f'\n\n###########\nNOW USING {smpl} SAMPLES in {valset} VALIDATION FOLD\n##############')
    smpl_results = {}

    valfold_mel = trainval[trainval.filenames.isin(mel_val[valset].filenames)]
    valfold_suz = trainval[trainval.filenames.isin(suz_val[valset].filenames)]
    validation = pd.concat([valfold_mel, valfold_suz])
    Xf_val = validation.filenames.tolist()
    yf_val = lenc.transform(validation.y)

    train = trainval[~trainval.filenames.isin(validation.filenames)]
    train_smpl = shuffle(train.sample(smpl, random_state=seed), random_state=seed)
    Xf_train, yf_train = train_smpl.filenames.tolist(), lenc.transform(train_smpl.y)

    print(f"Train: \n{pd.Series(yf_train).value_counts()}")
    print(f"Val: \n{pd.Series(yf_val).value_counts()}")
    print(f"Test: \n{pd.Series(yf_test).value_counts()}")    

    # cut_mosquito_model = Model(model.inputs, model.layers[-3].output)
    top_model = create_top_model(input_shape=cut_mosquito_model.output_shape[1:])
    Xf_xtracted_train, Xf_xtracted_val = extract_features_mosquitonet(Xf_train, yf_train, Xf_val, yf_val)
    yf_pred, h, modelC = train_model(Xf_train, yf_train, Xf_val, yf_val, Xf_xtracted_train, Xf_xtracted_val, 
                                     cut_mosquito_model, top_model, frozen_layers=frozen_layers)

    lb = LabelBinarizer()
    yf_pred_argmax = np.argmax(yf_pred, axis=1)
    cm = confusion_matrix(yf_test, yf_pred_argmax).astype(float)
    smpl_results['yf_test'] = yf_test
    smpl_results['yf_pred'] = yf_pred
    smpl_results['yf_pred_argmax'] = yf_pred_argmax
    smpl_results['nr_samples'] = smpl
    smpl_results['trial'] = trial
    smpl_results['frozen_layers'] = frozen_layers
    smpl_results['accuracy_score'] = accuracy_score(yf_test, yf_pred_argmax)
    smpl_results['balanced_accuracy_score'] = balanced_accuracy_score(yf_test, yf_pred_argmax)
    smpl_results['cm'] = cm
    #     smpl_results['classification_report'] = classification_report(yf_test, yf_pred_argmax)
    smpl_results['history'] = h.history

    dd.io.save(savepath, smpl_results)
    print('Done.')


# In[ ]:


samples = [100, 175, 250, 325, 400, 500, 1000, 2000, 4000, 6000, 8000]
froz_layerlist = [0, 19] #5, 15, 19]
trials = [0,1,2,3,4,5,6,7,8,9]

for trial in trials:
    for frozen_layers in froz_layerlist:
        for valset in range(3):
            for smpl in samples:
                myjob(trainval, mel_val, suz_val, Xf_test, yf_test, trial=trial, frozen_layers=frozen_layers, smpl=smpl, valset=valset)


# In[ ]:





# In[ ]:





# In[ ]:




