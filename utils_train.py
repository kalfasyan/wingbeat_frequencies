from utils import TEMP_DATADIR, SR, HOP_LEN, N_FFT, H_CUTOFF, F_S, L_CUTOFF, B_ORDER
import pandas as pd
import numpy as np
seed = 42
np.random.seed(seed)
import librosa
import cv2
from scipy import signal
from sklearn.utils import shuffle
import warnings
import logging
import math
from utils import crop_rec, TEMP_DATADIR
from wavhandler import Dataset, BASE_DIR
import time
import multiprocessing
import datetime
n_cpus = multiprocessing.cpu_count()

class ModelConfiguration(object):
    def __init__(self, model_setting=None, cnn_if_2d=None, target_names=None):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization,Input, LSTM, GRU
        from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
        from tensorflow.keras.optimizers import SGD
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.regularizers import l2
        # MODELS
        from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        from tensorflow.keras.applications.mobilenet import MobileNet
        from tensorflow.keras.applications.nasnet import NASNetLarge, NASNetMobile
        from tensorflow.keras.applications.vgg16 import VGG16
        from tensorflow.keras.applications.vgg19 import VGG19
        from tensorflow.keras.applications.xception import Xception

        super(ModelConfiguration, self).__init__()

        self.model_setting = model_setting
        self.cnn_if_2d = cnn_if_2d
        self.target_names = target_names

        if cnn_if_2d == 'DenseNet121':
            current_model = DenseNet121
        elif cnn_if_2d == 'DenseNet169':
            current_model = DenseNet169
        elif cnn_if_2d == 'DenseNet201':
            current_model = DenseNet201
        elif cnn_if_2d == 'InceptionResNetV2':
            current_model = InceptionResNetV2
        elif cnn_if_2d == 'VGG16':
            current_model = VGG16
        elif cnn_if_2d == 'VGG19':
            current_model = VGG19

        if model_setting in ['CONV2D', 'conv2d']:
            self.input_shape = (129, 120, 1)
        elif model_setting in ['gru','lstm','conv1d','CONV1D']:
            self.input_shape = (5000, 1)
        elif model_setting in ['conv1d_psd','CONV1D_psd','gru_psd','lstm_psd']:
            self.input_shape = (129, 1)
        else:
            raise ValueError('Wrong model_setting provided.')

        if model_setting in ['CONV2D', 'conv2d']:
            model = current_model(input_tensor = Input(shape = self.input_shape), 
                                classes = len(target_names), 
                                weights = None)
        elif model_setting in ['gru','lstm', 'gru_psd', 'lstm_psd']:
            model = Sequential()
            model.add(Conv1D(16, 3, activation='relu', input_shape=self.input_shape))
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
            if model_setting == 'gru' or model_setting == 'gru_psd':
                model.add(GRU(units= 128, return_sequences=True))
                model.add(GRU(units=128, return_sequences=False))
            if model_setting == 'lstm' or model_setting == 'lstm_psd':
                model.add(LSTM(units=128, return_sequences=True))
                model.add(LSTM(units=128, return_sequences=False))
            model.add(Dropout(0.5))
            model.add(Dense(len(target_names), activation='softmax'))
        elif model_setting == 'wavenet':
            model=Sequential()
            model.add(Conv1D(16, 3, activation='relu', input_shape=self.input_shape))
            for rate in (1,2,4,8)*2:
                model.add(Conv1D(filters=16*rate,
                                            kernel_size=3,
                                            padding="causal",
                                            activation="relu",
                                            dilation_rate=rate))
                model.add(Conv1D(filters=16*rate,
                                            kernel_size=3,
                                            padding="causal",
                                            activation="relu",
                                            dilation_rate=rate))
                model.add(BatchNormalization())
            model.add(MaxPooling1D(2))
            model.add(Conv1D(128, 2, activation='relu'))
            model.add(Conv1D(128, 2, activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(2))
            model.add(Conv1D(256, 2, activation='relu'))
            model.add(Conv1D(256, 2, activation='relu'))
            model.add(BatchNormalization())
            model.add(GlobalAveragePooling1D())
            model.add(Dropout(0.5))
            model.add(Dense(len(target_names), activation='softmax'))
        elif model_setting in ['conv1d','CONV1D','conv1d_psd','CONV1D_psd']:
            model = Sequential()
            model.add(Conv1D(16, 3, activation='relu', input_shape=self.input_shape))
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
            model.add(Dense(len(target_names), activation='softmax'))
        else:
            raise ValueError("Wrong model setting given.")

        self.config = model

class TrainConfiguration(object):
    """ Configuration for training procedures. Contains all settings """
    def __init__(self, dataset=None, setting='raw', model_name='test', batch_size=32, monitor='val_loss', 
                es_patience=7, rlr_patience=3, epochs=100):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation
        from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
        from tensorflow.keras.optimizers import SGD
        from tensorflow.keras.layers import BatchNormalization
        from sklearn.model_selection import train_test_split
        from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, CSVLogger
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras import utils

        super(TrainConfiguration, self).__init__()

        self.setting = setting
        self.model_name = model_name
        self.top_weights_path = TEMP_DATADIR + str(self.model_name) + '.h5'
        self.logfile = TEMP_DATADIR + str(self.model_name) + '.log'
        self.log_dir = f"{TEMP_DATADIR}/logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.batch_size = batch_size
        self.monitor = monitor
        self.es_patience = es_patience
        self.rlr_patience = rlr_patience
        self.epochs = epochs
        self.target_names = np.unique(dataset.target_classes)
        self.targets = len(self.target_names)
        self.callbacks_list = [ModelCheckpoint(monitor = self.monitor,
                                    filepath = self.top_weights_path,
                                    save_best_only = True,
                                    save_weights_only = False,
                                    verbose = 1),
                                EarlyStopping(monitor = self.monitor,
                                            patience = self.es_patience,
                                            verbose = 1),
                                ReduceLROnPlateau(monitor = self.monitor,
                                            factor = 0.1,
                                            patience = self.rlr_patience,
                                            verbose = 1),
                                # CSVLogger(filename = self.logfile),
                                TensorBoard(log_dir=self.log_dir, histogram_freq=1, profile_batch=0)]


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

def random_data_shift_simple(data, u, shift_pct=0.006, axis=0):
    if np.random.random() < u:
        data = np.roll(data, int(round(np.random.uniform(-(len(data)*shift_pct), (len(data)*shift_pct)))), axis=axis)
    return data

def train_generator(X_train, y_train, batch_size, target_names, setting='stft', preprocessing_train_stats=None):
    from tensorflow.keras import utils

    obj = create_settings_obj(setting)

    train_mean = preprocessing_train_stats.mean.squeeze() # squeeze is redundant now, but necessary if ImageDataGenerator is used
    # train_std = preprocessing_train_stats.std.squeeze()

    while True:
        for start in range(0, len(X_train), batch_size):
            x_batch = []
            y_batch = []
            
            end = min(start + batch_size, len(X_train))
            train_batch = X_train[start:end]
            labels_batch = y_train[start:end]
            
            for i in range(len(train_batch)):
                data, _ = librosa.load(train_batch[i], sr = SR)
                if 'increasing dataset' in train_batch[i].split('/'):
                    data = crop_rec(data)

#                 data = random_data_shift(data, u = .2)

                data = metamorphose(data, setting=setting, stg_obj=obj)
                # Center data
                data = data - train_mean
                # Expand dimensions
                data = np.expand_dims(data, axis = -1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            y_batch = utils.to_categorical(y_batch, len(target_names))
            
            yield x_batch, y_batch

def valid_generator(X_val, y_val, batch_size, target_names, setting='stft', preprocessing_train_stats=None):
    from tensorflow.keras import utils

    train_mean = preprocessing_train_stats.mean.squeeze() 
    obj = create_settings_obj(setting)
    while True:
        for start in range(0, len(X_val), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(X_val))
            test_batch = X_val[start:end]
            labels_batch = y_val[start:end]

            for i in range(len(test_batch)):
                data, _ = librosa.load(test_batch[i], sr = SR)
                if 'increasing dataset' in test_batch[i].split('/'):
                    data = crop_rec(data)

                data = metamorphose(data, setting=setting, stg_obj=obj)
                # Center data
                data = data - train_mean
                # Expand dimensions
                data = np.expand_dims(data, axis = -1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)

            y_batch = utils.to_categorical(y_batch, len(target_names))

            yield x_batch, y_batch


def metamorphose(data, setting='stft', stg_obj=None, img_sz=150):
    if setting=='stft':
        data = librosa.stft(data, n_fft = N_FFT, hop_length = HOP_LEN)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = librosa.amplitude_to_db(data)
        data = np.flipud(data).astype(float)
    elif setting == 'melspec':
        data = np.log10(librosa.feature.melspectrogram(data, sr=SR, hop_length=HOP_LEN))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = librosa.amplitude_to_db(data)
        data = np.flipud(data)
    elif setting == 'raw':
        return data
    elif setting == 'psd_dB':
        data = 10*np.log10(signal.welch(data, fs=F_S, window='hanning', nperseg=256, noverlap=128+64)[1])
        return data
    # elif setting=='gasf':
    #     data = stg_obj.fit_transform(data.reshape(1,-1)).squeeze()
    # elif setting=='gadf':
    #     data = stg_obj.fit_transform(data.reshape(1,-1)).squeeze()
    # elif setting=='mtf':
    #     data = stg_obj.fit_transform(data.reshape(1,-1)).squeeze()
    # elif setting=='rp':
    #     data = stg_obj.fit_transform(data.reshape(1,-1)).squeeze()
    #     data = cv2.resize(data,(img_sz,img_sz))
    else:
        raise ValueError("Wrong setting given.")
    return data

def create_settings_obj(setting='gasf', img_sz=150):
    # from pyts.image import GramianAngularField, RecurrencePlot, MarkovTransitionField
    if setting == 'stft' or setting == 'melspec' or setting == 'raw' or setting == 'psd_dB':
        obj = None
    # elif setting == 'gasf':
    #     obj = GramianAngularField(image_size=img_sz, method='summation')
    # elif setting == 'gadf':
    #     obj = GramianAngularField(image_size=img_sz, method='difference')
    # elif setting == 'mtf':
    #     obj = MarkovTransitionField(image_size=img_sz)
    # elif setting == 'rp':
    #     obj = RecurrencePlot(dimension=7, time_delay=3,
    #                 threshold='percentage_points',
    #                 percentage=30)
    else:
        raise ValueError("Wrong setting given.")
    return obj

def train_test_val_split(X,y, random_state=seed, verbose=1, test_size=0.10, val_size=0.2):
    from sklearn.model_selection import train_test_split
    X, y = shuffle(X, y, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, stratify=y_train, random_state=seed)
    if isinstance(X_train, np.ndarray):
        print("X_train shape: \t{}\nX_test shape:\t{}\nX_val shape:\t{}\n".format(X_train.shape, X_test.shape, X_val.shape))
    return X_train, X_test, X_val, y_train, y_test, y_val

def train_test_filenames(dataset, species, train_dates=[], test_dates=[], plot=False):
    dataset.read(species, loadmat=False)
    dataset.get_sensor_features()
    sub = dataset.df_features
    if plot:
        sub.groupby('datestr')['filenames'].count().plot(kind="bar")
    print(sub['datestr'].unique().tolist())

    test_fnames = sub[sub.datestr.isin(test_dates)].filenames
    if len(train_dates): # if train dates are given
        train_fnames = sub[sub.datestr.isin(train_dates)].filenames
    else:
        train_fnames = sub[~sub.datestr.isin(test_dates)].filenames

    print("{} train filenames, {} test filenames".format(train_fnames.shape[0], test_fnames.shape[0]))
    return train_fnames, test_fnames

def mosquito_data_split(splitting=None, dataset=None):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    assert splitting in ['random','randomcv','custom'], 'Wrong splitting argument passed.'
    # assert isinstance(data, Dataset), 'Pass a wavhandler.Dataset as data.'

    if splitting == 'random':
            dataset.read('Ae. aegypti', loadmat=False)
            x1 = dataset.filenames.sample(14800, random_state=seed)
            dataset.read('Ae. albopictus', loadmat=False)
            x2 = dataset.filenames.sample(14800, random_state=seed)
            dataset.read('An. arabiensis', loadmat=False)
            x3 = dataset.filenames.sample(14800, random_state=seed)
            dataset.read('An. gambiae', loadmat=False)
            x4 = dataset.filenames.sample(14800, random_state=seed)
            dataset.read('C. pipiens', loadmat=False)
            x5 = dataset.filenames.sample(14800, random_state=seed)
            dataset.read('C. quinquefasciatus', loadmat=False)
            x6 = dataset.filenames.sample(14800, random_state=seed)

            X = pd.concat([x1, x2, x3, x4, x5, x6], axis=0)
            y = X.apply(lambda x: x.split('/')[dataset.class_path_idx])

            le = LabelEncoder()
            y = le.fit_transform(y.copy())

            X,y = shuffle(X.tolist(),y.tolist(), random_state=seed)
            X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(X,y,test_size=0.13514, val_size=0.2)
    elif splitting in ['randomcv', 'custom']:
            # ### Ae. Aegypti
            x1_tr, x1_ts = train_test_filenames(dataset,'Ae. aegypti', test_dates=['20161213','20161212'])
            # ### Ae. albopictus
            x2_tr, x2_ts = train_test_filenames(dataset,'Ae. albopictus', test_dates=['20170103', '20170102'])
            # ### An. arabiensis
            x3_tr, x3_ts = train_test_filenames(dataset,'An. arabiensis', test_dates=['20170319','20170320',
                                                                                    '20170318','20170317'], 
                                                                        train_dates=['20170201','20170202', '20170203','20170204',
                                                                                    '20170205','20170206','20170131','20170130'])
            # ### An. gambiae
            x4_tr, x4_ts = train_test_filenames(dataset,'An. gambiae', test_dates=['20170110', '20170109']) 
            # ### Culex quinquefasciatus
            x5_tr, x5_ts = train_test_filenames(dataset,'C. quinquefasciatus', test_dates=['20161219']) 
            # ### Culex pipiens
            x6_tr, x6_ts = train_test_filenames(dataset,'C. pipiens', test_dates=['20161206', '20161205']) 

            x1_tr, x1_ts = x1_tr.sample(12800, random_state=seed), x1_ts.sample(2000, random_state=seed)
            x2_tr, x2_ts = x2_tr.sample(12800, random_state=seed), x2_ts.sample(2000, random_state=seed)
            x3_tr, x3_ts = x3_tr.sample(12800, random_state=seed), x3_ts.sample(2000, random_state=seed)
            x4_tr, x4_ts = x4_tr.sample(12800, random_state=seed), x4_ts.sample(2000, random_state=seed)
            x5_tr, x5_ts = x5_tr.sample(12800, random_state=seed), x5_ts.sample(2000, random_state=seed)
            x6_tr, x6_ts = x6_tr.sample(12800, random_state=seed), x6_ts.sample(2000, random_state=seed)

            # ## Creating TRAIN/VAL/TEST sets
            X_train = pd.concat([x1_tr, x2_tr, x3_tr, x4_tr, x5_tr, x6_tr], axis=0)
            X_test = pd.concat([x1_ts, x2_ts, x3_ts, x4_ts, x5_ts, x6_ts], axis=0)

            y_train = X_train.apply(lambda x: x.split('/')[dataset.class_path_idx])
            y_test = X_test.apply(lambda x: x.split('/')[dataset.class_path_idx])

            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.fit_transform(y_test)

            X_test = X_test.tolist()

            if splitting == 'randomcv':
                X_train,y_train = shuffle(X_train.tolist(),y_train.tolist(), random_state=seed)
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
            elif splitting == 'custom':
                import itertools
                from sklearn.model_selection import LeaveOneGroupOut

                # Creating DataFrame to sort data by date
                df = pd.DataFrame(X_train, columns=['filenames'])
                df['class'] = df['filenames'].apply(lambda x: x.split('/')[dataset.class_path_idx])
                df['wavnames'] = df['filenames'].apply(lambda x: x.split('/')[-1][:-4])
                df['date'] = df['wavnames'].apply(lambda x: pd.to_datetime(''.join(x.split('_')[0:2]),format='F%y%m%d%H%M%S'))
                df.sort_values(by='date', inplace=True)

                # For each mosquito we divide its data in 5 chunks
                classes = df['class'].unique().tolist()
                n_chunks = 5 # also number of folds
                class_chunks = {} # this is a dict with class as index and 5 lists in each class, 1 for each chunk
                for cl, sub in df.groupby('class'):
                    sub.reset_index(drop=True, inplace=True) # resetting index to have values from 1....sub.shape[0]
                    lst = list(range(0,sub.shape[0])) # this is basically the index
                    n = int(sub.shape[0] / n_chunks) # number of items in each chunk
                    inds_chunk = [np.array(lst[i:i + n]) for i in range(0, len(lst), n)] # splitting the index numbers (lst) in 5 chunks
                    class_chunks[cl] = [sub.iloc[inds_chunk[j]]['filenames'].tolist() for j in range(len(inds_chunk))] 

                # Now creating the actual folds for Cross validation 
                X_folds = {}
                y_folds = {}
                # For each fold, we add the corresponding chunks of each mosquito
                # This means that in FOLD-0 we will have the first chunk of each mosquito
                # in FOLD-1 we will have the second chunk of each mosquito etc..
                for i in range(n_chunks): 
                    X_folds[i] = [] # creating a list in which to add data of each mosquito
                    for c in classes: 
                        X_folds[i].extend(class_chunks[c][i]) # adding each mosquito chunk data in the list
                    y_folds[i] = pd.Series(X_folds[i]).apply(lambda x: x.split('/')[dataset.class_path_idx]).tolist() # targets

                # Creating groups to use the LeaveOneGroupOut of sklearn for combining
                # our X_folds so that 4 of them become train and 1 becomes val each time.
                groups = np.array(range(len(X_folds)))

                logo = LeaveOneGroupOut()
                X_train, X_val, y_train, y_val = [], [], [], []
                for train_index, val_index in logo.split(X_folds, y_folds, groups):
                    train_groups = [X_folds.get(key) for key in train_index]
                    train_groups_untd = pd.Series(list(itertools.chain.from_iterable(train_groups)))

                    X_train.append(train_groups_untd.tolist())
                    train_labels = train_groups_untd.apply(lambda x: x.split('/')[dataset.class_path_idx]).tolist()
                    y_train.append(list(le.fit_transform(train_labels)))
                    
                    val_group = pd.Series(X_folds[val_index[0]])
                    X_val.append(val_group.tolist())
                    val_labels = val_group.apply(lambda x: x.split('/')[dataset.class_path_idx]).tolist()
                    y_val.append(list(le.fit_transform(val_labels)))
    else:
        raise NotImplementedError('NOT IMPLEMENTED YED')
    return X_train, X_val, X_test, y_train, y_val, y_test

def calculate_train_statistics(X_train=None, setting=None):
    from wavhandler import make_df_parallel
    # from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from collections import OrderedDict

    # train_stats = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    train_stats = OrderedDict()

    train_matrix = make_df_parallel(names=X_train, setting=setting).values
    if setting == 'stft':
        train_matrix = train_matrix.reshape((len(X_train),129,120,1))
    elif setting == 'raw':
        train_matrix = train_matrix.reshape((len(X_train),5000,1))
    elif setting == 'psd_dB':
        train_matrix = train_matrix.reshape((len(X_train),129,1))
    else:
        raise ValueError('Wrong setting provided.')

    # train_stats.fit(train_matrix)
    train_stats.mean = np.mean(train_matrix, axis=(0,1))

    return train_stats

def train_model_ml(dataset=None, model_setting=None, splitting=None, data_setting=None,
                    x_train=None, y_train=None,
                    x_val=None, y_val=None,
                    x_test=None, y_test=None, flag=None):
    """
    Used to train ML models on wingbeat data, depending on the splitting method chosen (random/randomcv or custom).
    Models used so far: KNeighborsClassifier, RandomForestClassifier, XGBXClassifier
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
    from xgboost import XGBClassifier
    from wavhandler import make_df_parallel
    from joblib import dump, load
    from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report, make_scorer, log_loss

    # Defining the chosen estimator
    if model_setting.startswith('knn'):
        estimator = KNeighborsClassifier(n_neighbors=11, weights='uniform',metric='manhattan', n_jobs=min(8,n_cpus))
    elif model_setting.startswith('randomforest'):
        estimator = RandomForestClassifier(bootstrap=True, max_depth=None,
                                            min_samples_leaf=3, min_samples_split=8,
                                            max_features='auto', criterion='gini',
                                            n_estimators=450, n_jobs=min(8,n_cpus),
                                            random_state=seed, verbose=True)
    elif model_setting.startswith('xgboost'):
        estimator = XGBClassifier(max_depth=4,
                                    n_estimators=450,
                                    learning_rate=0.3,
                                    gamma=0.5,
                                    random_state=seed,
                                    seed=seed,
                                    verbose=True)
    else:
        raise NotImplementedError('Not implemented yet.')

    # Training and reporting cross-validation and test results by saving them to a text file.
    if splitting in ['random', 'randomcv']:
        cvfolds = 5
        cv_results = cross_validate(estimator, x_train, y_train, cv=cvfolds, 
                                    return_estimator=True, 
                                    return_train_score=True, 
                                    scoring=make_scorer(balanced_accuracy_score),
                                    verbose=1, 
                                    n_jobs=min(8, n_cpus)) 

        y_preds = [cv_results['estimator'][i].predict(x_test) for i in range(cvfolds)]
        y_pred_probas = [cv_results['estimator'][i].predict_proba(x_test) for i in range(cvfolds)]

        cms = [confusion_matrix(y_test, y_preds[i]) for i in range(cvfolds)]
        b_accs = [balanced_accuracy_score(y_test, y_preds[i]) for i in range(cvfolds)]
        logloss = [log_loss(y_test, y_pred_probas[i]) for i in range(cvfolds)]
        clf_reports = [classification_report(y_test, y_preds[i], target_names=dataset.target_classes) for i in range(cvfolds)]

        mean_train_score = np.mean(cv_results['train_score'])
        mean_val_score = np.mean(cv_results['test_score'])
        mean_test_score = np.mean(b_accs)

        for i in range(cvfolds):
            with open(f'temp_data/{splitting}_{data_setting}_{model_setting}_{flag}_results.txt', "a+") as resultsfile:
                if i == 0:
                    resultsfile.write(f'mean_train_score: {mean_train_score},'
                                        f'mean_val_score: {mean_val_score},'
                                        f'mean_test_score: {mean_test_score}\n') 
                resultsfile.write(f'\n\n\t\tFOLD #: {i}\n '
                                    f'train_score: {cv_results["train_score"][i]},' 
                                    f'val_score: {cv_results["test_score"][i]},' 
                                    f'balanced_accuracy_on_test: {b_accs[i]}\n,' 
                                    f'log_loss_on_test: {logloss[i]}\n,' 
                                    f'confusion_matrix:\n{cms[i]}\n' 
                                    f'classification_report:\n{clf_reports[i]}\n')
    elif splitting == 'custom':
        estimator.fit(x_train,y_train)
        return estimator
    else:
        raise ValueError('Wrong splitting method provided.')

    return estimator

def train_model_dl(dataset=None, model_setting=None, splitting=None, data_setting=None, cnn_if_2d=None,
                X_train=None, y_train=None,
                X_val=None, y_val=None,
                X_test=None, y_test=None, flag=None):
    from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report, make_scorer, log_loss


    print(f'processing: {cnn_if_2d}_{flag}')
    traincf = TrainConfiguration(dataset=dataset, setting=data_setting, model_name=f'{splitting}_{data_setting}_{model_setting}_{cnn_if_2d}_{flag}')
    model = ModelConfiguration(model_setting=model_setting, cnn_if_2d=cnn_if_2d, target_names=traincf.target_names).config

    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

    train_stats = calculate_train_statistics(X_train=X_train, setting=data_setting)
    # Actual training
    h = model.fit_generator(train_generator(X_train, y_train, 
                                        batch_size=traincf.batch_size,
                                        target_names=traincf.target_names,
                                        setting=traincf.setting,
                                        preprocessing_train_stats=train_stats),
                        steps_per_epoch = int(math.ceil(float(len(X_train)) / float(traincf.batch_size))),
                        epochs=traincf.epochs,
                        validation_data = valid_generator(X_val, y_val,
                                                            batch_size=traincf.batch_size,
                                                            target_names=traincf.target_names,
                                                            setting=traincf.setting,
                                                            preprocessing_train_stats=train_stats),
                        validation_steps=int(math.ceil(float(len(X_test))/float(traincf.batch_size))),
                        callbacks=traincf.callbacks_list,
                        use_multiprocessing=False,
                        workers=1,
                        max_queue_size=32)

    train_loss = h.history['loss']
    train_score = h.history['accuracy']
    val_loss = h.history['val_loss']
    val_score = h.history['val_accuracy']
    lr = h.history['lr']

    # LOADING TRAINED WEIGHTS
    model.load_weights(traincf.top_weights_path)

    y_pred = model.predict_generator(valid_generator(X_test, 
                                                    y_test, 
                                                    batch_size=traincf.batch_size, 
                                                    setting=traincf.setting, 
                                                    target_names=traincf.target_names,
                                                    preprocessing_train_stats=train_stats),
            steps = int(math.ceil(float(len(X_test)) / float(traincf.batch_size))))
    bacc = balanced_accuracy_score(np.array(y_test), np.argmax(y_pred, axis=1))
    cm = confusion_matrix(np.array(y_test), np.argmax(y_pred, axis=1))
    test_loss = log_loss(y_test, y_pred)
    clf_report = classification_report(y_test, np.argmax(y_pred, axis=1))

    # Saving results
    with open(f'temp_data/{splitting}_{data_setting}_{model_setting}_results.txt', "a+") as resultsfile:
        resultsfile.write(f'\n\n\t\tFOLD #: {flag}\n '
                            f'train_score: {train_score}\n'
                            f'train_loss: {train_loss}\n' 
                            f'val_score: {val_score}\n' 
                            f'val_loss: {val_loss}\n'
                            f'balanced_accuracy_on_test: {bacc}\n' 
                            f'log_loss_on_test: {test_loss}\n' 
                            f'learning_rate: {lr}\n'
                            f'confusion_matrix:\n{cm}\n' 
                            f'classification_report:\n{clf_report}\n')

    import deepdish as dd
    results = h.history
    results['train_score'] = train_score
    results['train_loss'] = train_loss
    results['val_score'] = val_score
    results['val_loss'] = val_loss
    results['balanced_acc_test'] = bacc
    results['logloss_test'] = test_loss
    results['learning_rate'] = lr
    results['y_pred'] = y_pred
    results['y_test'] = y_test
    
    # Save it under the form of a json file
    dd.io.save(f'temp_data/{splitting}_{data_setting}_{model_setting}_results.h5', {f'results_{flag}': results})
