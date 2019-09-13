%reset -f
from wavhandler import *
import pandas as pd
import numpy as np
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

data = Dataset('Wingbeats')
data.read(data="all", setting='read', loadmat=False, labels='nr')
data.split()

# data1 = Dataset('Leafminers')
# data1.read(data=data1.target_classes[0], setting='read', loadmat=False, labels='nr')

# data2 = Dataset('LG')
# data2.read(data=data2.target_classes[0], setting='read', loadmat=False, labels='nr')

# data3 = Dataset('LG')
# data3.read(data=data3.target_classes[1], setting='read', loadmat=False, labels='nr')

# data4 = Dataset('Pcfruit')
# data4.read(data=data4.target_classes[1], setting='read', loadmat=False, labels='nr')

# # data1.clean(plot=False)
# # data2.clean(plot=False)
# # data3.clean(plot=False)
# # data4.clean(plot=False)

# data = pd.DataFrame()
# data['filenames'] = pd.concat([data1.filenames, data2.filenames, data3.filenames, data4.filenames], axis=0).reset_index(drop=True)
# data['y'] = data.filenames.apply(lambda x: x.split('/')[6])

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# data['y'] = le.fit_transform(data['y'])


X_train, X_test, X_val, y_train, y_test, y_val = data.X_train.tolist(), \
                                                data.X_test.tolist(), \
                                                data.X_val.tolist(), \
                                                data.y_train.tolist(), \
                                                data.y_test.tolist(), \
                                                data.y_val.tolist() 

model_name='mosquitoes_final_raw' 
setting='raw'
top_weights_path = TEMP_DATADIR + str(model_name) + '.h5'
targets = len(np.unique(data.y))
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

target_names = list(np.unique(data.y))

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

model.load_weights(top_weights_path)
loss, acc = model.evaluate_generator(valid_generator(X_test, 
                                                    y_test, 
                                                    batch_size=batch_size, 
                                                    setting=setting, 
                                                    target_names=target_names),
        steps = int(math.ceil(float(len(X_test)) / float(batch_size))))

print('loss', loss)
print('Test accuracy:', acc)