#!/usr/bin/env python
# coding: utf-8

# In[126]:


# get_ipython().run_line_magic('reset', '-f')
from wavhandler import *
from utils_train import train_test_val_split
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# In[127]:


data = Dataset('Wingbeats')
data.target_classes


# ### Ae. Aegypti

# In[128]:


data1 = Dataset('Wingbeats')
data1.read('Ae. aegypti', loadmat=False)
data1.get_sensor_features();


# In[129]:


# data1.df_features.date.hist(xrot=45)


# In[130]:


sub = data1.df_features
inds_1 = sub[sub.date < '2017-01-01'].index


# ### Ae. albopictus

# In[131]:


data2 = Dataset('Wingbeats')
data2.read('Ae. albopictus', loadmat=False)
data2.get_sensor_features(temp_humd=False)


# In[132]:


# data2.df_features.date.hist(xrot=45)


# In[133]:


sub = data2.df_features
inds_2 = sub[sub.date < '2017-01-12'].index


# ### An. arabiensis

# In[134]:


data3 = Dataset('Wingbeats')
data3.read('An. arabiensis', loadmat=False)

data3.get_sensor_features(temp_humd=False)


# In[135]:


# data3.df_features.date.hist(xrot=45)


# In[136]:


sub = data3.df_features
inds_3 = sub[sub.date > '2017-03-17'].index


# ### An. gambiae

# In[137]:


data4 = Dataset('Wingbeats')
data4.read('An. gambiae', loadmat=False)

data4.get_sensor_features(temp_humd=False)


# In[138]:


# data4.df_features.date.hist(xrot=45)


# In[139]:


sub = data4.df_features
inds_4 = sub[sub.date < '2017-01-11'].index


# ### Culex quinquefasciatus

# In[140]:


data5 = Dataset('Wingbeats')
data5.read('C. quinquefasciatus', loadmat=False)

data5.get_sensor_features(temp_humd=False)


# In[141]:


sub = data5.df_features
# inds_5 = sub[sub.date < '2016-12-20'].index
print(data5.df_features.query('date_day == 20').shape)
inds_5 = data5.df_features.query('date_day == 20').sample(2500).index
index_drop = data5.filenames[ (~data5.filenames.index.isin(inds_5))  & (data5.df_features.date_day == 20) ].index

data5.filenames.drop(index=index_drop, inplace=True)
data5.y.drop(index=index_drop, inplace=True)


# In[142]:


data5.get_sensor_features()
# data5.df_features.date.hist(xrot=45)


# ### Culex pipiens

# In[143]:


data6 = Dataset('Wingbeats')
data6.read('C. pipiens', loadmat=False)

data6.get_sensor_features(temp_humd=False)


# In[144]:


# data6.df_features.date.hist(xrot=45)


# In[145]:


sub = data6.df_features
inds_6 = sub[sub.date < '2017-01-01'].index


# ## Creating TRAIN/VAL/TEST sets

# In[146]:


def sub_test(dataset, inds):
    X = dataset.filenames.loc[~dataset.filenames.index.isin(inds)]
    y = dataset.y.loc[~dataset.y.index.isin(inds)]
    X_test = dataset.filenames.loc[dataset.filenames.index.isin(inds)]
    y_test = dataset.y.loc[dataset.y.index.isin(inds)]
    return X, y, X_test, y_test

x1, y1, xt1, yt1 = sub_test(data1, inds_1)
x2, y2, xt2, yt2 = sub_test(data2, inds_2)
x3, y3, xt3, yt3 = sub_test(data3, inds_3)
x4, y4, xt4, yt4 = sub_test(data4, inds_4)
x5, y5, xt5, yt5 = sub_test(data5, inds_5)
x6, y6, xt6, yt6 = sub_test(data6, inds_6)


# In[147]:


X_test = pd.concat([xt1,xt2,xt3,xt4,xt5,xt6], axis=0).tolist()
y_test = pd.concat([yt1,yt2,yt3,yt4,yt5,yt6], axis=0).tolist()


# In[148]:


X = pd.concat([x1,x2,x3,x4,x5,x6], axis=0).tolist()
y = pd.concat([y1,y2,y3,y4,y5,y6], axis=0).tolist()

df = pd.DataFrame({"X": X, "y":y})
df.drop(index=df[df['y'] == 'Ae. aegypti'].sample(38345).index.values, inplace=True)
X,y = df.X.values, df.y.values

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
y_test = le.fit_transform(y_test)


# In[149]:


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
X,y = shuffle(X,y, random_state=0)


# In[150]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)


# In[151]:


pd.Series(y_train).value_counts()
# le.inverse_transform(y_train)


# # Modelling

# In[153]:


from utils_train import *
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.utils import to_categorical
from wavhandler import *
import math


# In[154]:


model_name='mosquitoes_validation' 
setting='raw'
top_weights_path = TEMP_DATADIR + str(model_name) + '.h5'
targets = 6
batch_size = 32

# Build the Neural Network
model = Sequential()

model.add(Conv1D(16, 3, activation='relu', input_shape=(5000, 1)))
model.add(Conv1D(16, 3, activation='relu'))
model.add(BatchNormalization())

model.add(Conv1D(32, 3, activation='relu'))
model.add(Conv1D(32, 3, activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling1D(2))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling1D(2))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling1D(2))
model.add(Conv1D(256, 3, activation='relu'))
model.add(Conv1D(256, 3, activation='relu'))
model.add(BatchNormalization())
model.add(GlobalAveragePooling1D())

model.add(Dropout(0.5))
model.add(Dense(targets, activation='softmax'))

model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

callbacks_list = [ModelCheckpoint(top_weights_path, monitor = 'val_acc', verbose = 1, save_best_only = True, save_weights_only = True),
    EarlyStopping(monitor = 'val_acc', patience = 6, verbose = 1),
    ReduceLROnPlateau(monitor = 'val_acc', factor = 0.1, patience = 3, verbose = 1),
    CSVLogger('model_' + str(model_name) + '.log')]

target_names = list(np.unique(y))


# In[155]:


model.fit_generator(train_generator(X_train, y_train, batch_size=batch_size,
                                   target_names=target_names,
                                   setting=setting),
                   steps_per_epoch = int(math.ceil(float(len(X_train)) / float(batch_size))),
                   epochs=100,
                   validation_data = valid_generator(X_val, y_val,
                                                    batch_size=batch_size,
                                                    target_names=target_names,
                                                    setting=setting),
                    validation_steps=int(math.ceil(float(len(X_test))/float(batch_size))),
                    callbacks = callbacks_list)


# In[29]:


model.summary()


# In[77]:


model.load_weights(top_weights_path)
y_pred = model.predict_generator(valid_generator(X_test, 
                                                    y_test, 
                                                    batch_size=batch_size, 
                                                    setting=setting, 
                                                    target_names=target_names),
        steps = int(math.ceil(float(len(X_test)) / float(batch_size))))


# In[79]:


import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
x = confusion_matrix(np.array(y_test), np.argmax(y_pred, axis=1))

plt.figure(figsize=(16,12))
sns.set(font_scale=1.2)
ticks = ['Ae. aegypti',' Ae. albopictus','An. arabiensis','An. gambiae','C. pipiens','C. quinquefasciatus']
ticks_short = ['Ae. aeg','Ae. alb','An. arab','An. gambiae','C. pip','C. quin']
sns.heatmap(x, annot=True, fmt='.0f', xticklabels=ticks, yticklabels=ticks_short)


# In[ ]:


model.load_weights(top_weights_path)
loss, acc = model.evaluate_generator(valid_generator(X_test, 
                                                    y_test, 
                                                    batch_size=batch_size, 
                                                    setting=setting, 
                                                    target_names=target_names),
        steps = int(math.ceil(float(len(X_test)) / float(batch_size))))

print('loss', loss)
print('Test accuracy:', acc)

