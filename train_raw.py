#%%
import pandas as pd
import numpy as np
from wavhandler import *
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#get_ipython().run_line_magic('matplotlib', 'inline')


#%%
data1 = Dataset('Wingbeats')
data1.load(only_names=True, nr_signals=5000, text_labels=True);
data2 = Dataset('LG')
data2.load(only_names=True, text_labels=True);


#%%
data_X_names = data1.filenames + data2.filenames
data_y = data1.y + data2.y


#%%
df = make_df_parallel(names=data_X_names, setting='read').T


#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

y = LabelEncoder().fit_transform(y=data_y)

X_train, X_test, y_train, y_test = train_test_split(df.values, y, 
                                                    test_size=0.10, 
                                                    shuffle=True, 
                                                    random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                  test_size=0.2, 
                                                  random_state=0)
print("Train shape: \t{}, \nTest shape: \t{}, \nValid shape: \t{}".format(X_train.shape, X_test.shape, X_val.shape))


#%%
import keras
from sklearn.preprocessing import LabelEncoder

nr_classes = len(np.unique(data_y))

# Convert label to onehot
y_train = keras.utils.to_categorical(y_train, num_classes=nr_classes)
y_val = keras.utils.to_categorical(y_val, num_classes=nr_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=nr_classes)

X_train = np.expand_dims(X_train, axis=2)
X_val = np.expand_dims(X_val, axis=2)
X_test = np.expand_dims(X_test, axis=2)


#%%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

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
model.add(Dense(nr_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


#%%
model_name = 'raw_BIG'
top_weights_path = TEMP_DATADIR + '/model_' + str(model_name) + '.h5'


#%%
callbacks_list = [ModelCheckpoint(top_weights_path, monitor = 'val_acc', verbose = 1, save_best_only = True, save_weights_only = True),
    EarlyStopping(monitor = 'val_acc', patience = 6, verbose = 1),
    ReduceLROnPlateau(monitor = 'val_acc', factor = 0.1, patience = 3, verbose = 1),
    CSVLogger('model_' + str(model_name) + '.log')]


#%%
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data = [X_val, y_val], callbacks = callbacks_list)


#%%
model.load_weights(top_weights_path)
loss, acc = model.evaluate(X_test, y_test, batch_size=16)

#print('loss', loss)
print('Test accuracy:', acc)


#%%
from keras.models import model_from_yaml
# serialize model to YAML
model_yaml = model.to_yaml()
with open(TEMP_DATADIR + model_name + ".yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
model.save_weights(TEMP_DATADIR + model_name + "_weights.h5")


