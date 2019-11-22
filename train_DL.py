from wavhandler import *
import numpy as np
import sys
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from utils_train import train_test_val_split, TrainConfiguration, test_inds, test_days, train_generator, valid_generator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization,Input, LSTM, GRU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.densenet import DenseNet121

np.random.seed(42)

def train_test_filenames(dataset, species, train_dates=[], test_dates=[]):
    dataset.read(species, loadmat=False)
    dataset.get_sensor_features()
    sub = dataset.df_features
    sub.groupby('datestr')['filenames'].count().plot(kind="bar")
    print(sub['datestr'].unique().tolist())

    test_fnames = sub[sub.datestr.isin(test_dates)].filenames
    if len(train_dates): # if train dates are given
        train_fnames = sub[sub.datestr.isin(train_dates)].filenames
    else:
        train_fnames = sub[~sub.datestr.isin(test_dates)].filenames

    print("{} train filenames, {} test filenames".format(train_fnames.shape[0], test_fnames.shape[0]))
    return train_fnames, test_fnames

splitting = sys.argv[1]
data_setting = sys.argv[2]
model_setting = sys.argv[3]

assert splitting in ['random','improved'], "Wrong splitting method given."
assert data_setting in ['raw','stft', "Wrong data settting given."]
assert model_setting in ['lstm','gru','LSTM','GRU','CONV1D','CONV2D','conv1d','conv2d']

data = Dataset('Wingbeats')
print(data.target_classes)

print(f'SPLITTING DATA {splitting}')
if splitting == 'random':
        data.read('Ae. aegypti', loadmat=False)
        x1 = data.filenames.sample(14800)
        data.read('Ae. albopictus', loadmat=False)
        x2 = data.filenames.sample(14800)
        data.read('An. arabiensis', loadmat=False)
        x3 = data.filenames.sample(14800)
        data.read('An. gambiae', loadmat=False)
        x4 = data.filenames.sample(14800)
        data.read('C. pipiens', loadmat=False)
        x5 = data.filenames.sample(14800)
        data.read('C. quinquefasciatus', loadmat=False)
        x6 = data.filenames.sample(14800)

        X = pd.concat([x1, x2, x3, x4, x5, x6], axis=0)
        y = X.apply(lambda x: x.split('/')[len(BASE_DIR.split('/'))])

        text_y = y
        le = LabelEncoder()
        y = le.fit_transform(y.copy())

        X,y = shuffle(X.tolist(),y.tolist(), random_state=0)
        X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(X,y,test_size=0.13514, val_size=0.2)
elif splitting == 'improved':
        # ### Ae. Aegypti
        x1_tr, x1_ts = train_test_filenames(data,'Ae. aegypti', test_dates=['20161213','20161212'])
        # ### Ae. albopictus
        x2_tr, x2_ts = train_test_filenames(data,'Ae. albopictus', test_dates=['20170103', '20170102'])
        # ### An. arabiensis
        x3_tr, x3_ts = train_test_filenames(data,'An. arabiensis', test_dates=['20170319','20170320',
                                                                        '20170318','20170317'], train_dates=['20170201','20170202', '20170203','20170204',
                                                                                                                '20170205','20170206','20170131','20170130'])
        # ### An. gambiae
        x4_tr, x4_ts = train_test_filenames(data,'An. gambiae', test_dates=['20170110', '20170109']) 
        # ### Culex quinquefasciatus
        x5_tr, x5_ts = train_test_filenames(data,'C. quinquefasciatus', test_dates=['20161219']) 
        # ### Culex pipiens
        x6_tr, x6_ts = train_test_filenames(data,'C. pipiens', test_dates=['20161206', '20161205']) 

        x1_tr, x1_ts = x1_tr.sample(12800), x1_ts.sample(2000)
        x2_tr, x2_ts = x2_tr.sample(12800), x2_ts.sample(2000)
        x3_tr, x3_ts = x3_tr.sample(12800), x3_ts.sample(2000)
        x4_tr, x4_ts = x4_tr.sample(12800), x4_ts.sample(2000)
        x5_tr, x5_ts = x5_tr.sample(12800), x5_ts.sample(2000)
        x6_tr, x6_ts = x6_tr.sample(12800), x6_ts.sample(2000)

        # ## Creating TRAIN/VAL/TEST sets
        X_train = pd.concat([x1_tr, x2_tr, x3_tr, x4_tr, x5_tr, x6_tr], axis=0)
        X_test = pd.concat([x1_ts, x2_ts, x3_ts, x4_ts, x5_ts, x6_ts], axis=0)

        y_train = X_train.apply(lambda x: x.split('/')[len(BASE_DIR.split('/'))])
        y_test = X_test.apply(lambda x: x.split('/')[len(BASE_DIR.split('/'))])

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.fit_transform(y_test)

        X_test = X_test.tolist()

        X_train,y_train = shuffle(X_train.tolist(),y_train.tolist(), random_state=0)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# PRINTING THE CLASS BALANCE
keys = pd.Series(le.inverse_transform(y_train)).value_counts().index.tolist()
values = pd.Series(y_train).value_counts().index.tolist()
mapping = dict(zip(keys, values))
print(sorted(mapping.items(), key=lambda x: x[1]))
vcounts = pd.Series(y_train).value_counts()
vcounts.index = mapping.keys()
print(vcounts)

# MODELLING

traincf = TrainConfiguration(X=X_train, y=y_train, setting=data_setting, model_name=f'{splitting}_{data_setting}_{model_setting}')
targets = 6

print(f'### MODEL NAME ==== {traincf.model_name} ####')

if model_setting in ['gru','GRU','LSTM','lstm']:
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
    model.add(MaxPooling1D(2))
    if model_setting == 'gru':
        model.add(GRU(units= 128, return_sequences=True))
        model.add(GRU(units=128, return_sequences=False))
    if model_setting == 'lstm':
        model.add(LSTM(units=128, return_sequences=True))
        model.add(LSTM(units=128, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(targets, activation='softmax'))
elif model_setting == 'CONV2D':
    current_model = DenseNet121
    model = current_model(input_tensor = Input(shape = (129, 120, 1)), 
                        classes = len(traincf.target_names), 
                        weights = None)
elif model_setting == 'CONV1D':
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
else:
    raise ValueError("Wrong model setting given.")

model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

callbacks_list = traincf.callbacks_list

model.fit_generator(train_generator(X_train, y_train, batch_size=traincf.batch_size,
                                   target_names=traincf.target_names,
                                   setting=traincf.setting),
                   steps_per_epoch = int(math.ceil(float(len(X_train)) / float(traincf.batch_size))),
                   epochs=traincf.epochs,
                   validation_data = valid_generator(X_val, y_val,
                                                    batch_size=traincf.batch_size,
                                                    target_names=traincf.target_names,
                                                    setting=traincf.setting),
                    validation_steps=int(math.ceil(float(len(X_test))/float(traincf.batch_size))),
                    callbacks = traincf.callbacks_list)

model.load_weights(traincf.top_weights_path)
y_pred = model.predict_generator(valid_generator(X_test, 
                                                    y_test, 
                                                    batch_size=traincf.batch_size, 
                                                    setting=traincf.setting, 
                                                    target_names=traincf.target_names),
        steps = int(math.ceil(float(len(X_test)) / float(traincf.batch_size))))

from sklearn.metrics import confusion_matrix
x = confusion_matrix(np.array(y_test), np.argmax(y_pred, axis=1))
print(x)

model.load_weights(traincf.top_weights_path)
loss, acc = model.evaluate_generator(valid_generator(X_test, 
                                                    y_test, 
                                                    batch_size=traincf.batch_size, 
                                                    setting=traincf.setting, 
                                                    target_names=traincf.target_names),
        steps = int(math.ceil(float(len(X_test)) / float(traincf.batch_size))))

print('loss', loss)
print('Test accuracy:', acc)

print(f'### MODEL NAME ==== {traincf.model_name} ####')

from sklearn.metrics import balanced_accuracy_score
print('Balanced accuracy:')
print(balanced_accuracy_score(np.array(y_test), np.argmax(y_pred, axis=1)))
