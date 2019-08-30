from utils import TEMP_DATADIR, SR, HOP_LEN, N_FFT, H_CUTOFF, F_S, L_CUTOFF, B_ORDER
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
# from pyts.image import GramianAngularField, RecurrencePlot, MarkovTransitionField
import cv2
from scipy import signal
from sklearn.metrics import confusion_matrix
# from keras.utils import np_utils
# from keras.preprocessing import image
from sklearn.utils import shuffle
import warnings
import logging
import math
from utils import crop_rec
from keras.utils import np_utils

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

def train_generator(X_train, y_train, batch_size, target_names, setting='stft'):
    obj = create_settings_obj(setting)
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

                data = np.expand_dims(data, axis = -1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            
            y_batch = np_utils.to_categorical(y_batch, len(target_names))
            
            yield x_batch, y_batch

def valid_generator(X_val, y_val, batch_size, target_names, setting='stft'):
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

                data = np.expand_dims(data, axis = -1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)

            y_batch = np_utils.to_categorical(y_batch, len(target_names))

            yield x_batch, y_batch


def train_generator2(X_train, batch_size, target_names, setting='stft'):
    obj = create_settings_obj(setting)
    while True:
        for start in range(0, len(X_train), batch_size):
            x_batch = []
            
            end = min(start + batch_size, len(X_train))
            train_batch = X_train[start:end]
            
            for i in range(len(train_batch)):
                data, _ = librosa.load(train_batch[i], sr = SR)
                if 'increasing dataset' in train_batch[i].split('/'):
                    data = crop_rec(data)

#                 data = random_data_shift(data, u = .2)

                data = metamorphose(data, setting=setting, stg_obj=obj)
                data = data[2:,2:]

                data = np.expand_dims(data, axis = -1)

                x_batch.append(data)

            x_batch = np.array(x_batch, np.float32)

            yield x_batch, x_batch

def valid_generator2(X_val, batch_size, target_names, setting='stft'):
    obj = create_settings_obj(setting)
    while True:
        for start in range(0, len(X_val), batch_size):
            x_batch = []

            end = min(start + batch_size, len(X_val))
            test_batch = X_val[start:end]

            for i in range(len(test_batch)):
                data, _ = librosa.load(test_batch[i], sr = SR)
                if 'increasing dataset' in test_batch[i].split('/'):
                    data = crop_rec(data)

                data = metamorphose(data, setting=setting, stg_obj=obj)
                
                data = data[2:,2:]

                data = np.expand_dims(data, axis = -1)

                x_batch.append(data)

            x_batch = np.array(x_batch, np.float32)

            yield x_batch, x_batch


def metamorphose(data, setting='stft', stg_obj=None, img_sz=150):
    if setting=='stft':
        data = librosa.stft(data, n_fft = N_FFT, hop_length = HOP_LEN)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = librosa.amplitude_to_db(data)
        data = np.flipud(data)
    elif setting == 'melspec':
        data = np.log10(librosa.feature.melspectrogram(data, sr=SR, hop_length=HOP_LEN))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = librosa.amplitude_to_db(data)
        data = np.flipud(data)
    elif setting=='gasf':
        data = stg_obj.fit_transform(data.reshape(1,-1)).squeeze()
    elif setting=='gadf':
        data = stg_obj.fit_transform(data.reshape(1,-1)).squeeze()
    elif setting=='mtf':
        data = stg_obj.fit_transform(data.reshape(1,-1)).squeeze()
    elif setting=='rp':
        data = stg_obj.fit_transform(data.reshape(1,-1)).squeeze()
        data = cv2.resize(data,(img_sz,img_sz))
    else:
        raise ValueError("Wrong setting given.")
    return data

def create_settings_obj(setting='gasf', img_sz=150):
    if setting == 'stft':
        obj = None
    elif setting == 'melspec':
        obj = None
    elif setting == 'gasf':
        obj = GramianAngularField(image_size=img_sz, method='summation')
    elif setting == 'gadf':
        obj = GramianAngularField(image_size=img_sz, method='difference')
    elif setting == 'mtf':
        obj = MarkovTransitionField(image_size=img_sz)
    elif setting == 'rp':
        obj = RecurrencePlot(dimension=7, time_delay=3,
                    threshold='percentage_points',
                    percentage=30)
    else:
        raise ValueError("Wrong setting given.")
    return obj

def make_classification_ml(X,y, clf_name='xgboost', undersampling=True, confmat=True, featimpt=True):
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    import xgboost
    from utils import get_classifier

    X, y = shuffle(X, y, random_state=0)
    if undersampling:
        from imblearn.under_sampling import RandomUnderSampler
        ros = RandomUnderSampler(random_state=0)
        ros.fit(X,y)
        X, y = ros.fit_resample(X,y)
        print('After undersampling: \n{}\n'.format(pd.DataFrame(y).iloc[:,0].value_counts()))
    else:
        print('Class balance: \n{}\n'.format(pd.DataFrame(y).iloc[:,0].value_counts()))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,  stratify=y, random_state=0)
    classifier = get_classifier(clf_name)
    #xgboost.XGBClassifier(n_estimators=150, learning_rate=0.2, n_jobs=-1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    ac = accuracy_score(y_test, y_pred)
    print("Name: %s, ac: %f" % ('model', ac))
    if confmat:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='g')
        plt.show()
    if featimpt:
        feature_importances = pd.DataFrame(classifier.feature_importances_,
                                    columns=['importance']).sort_values('importance', ascending=False)
        print(feature_importances.head())
    return classifier

def make_classification_conv1d(X,y, model_name='test_', setting='raw'):
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
    from keras.optimizers import SGD
    from keras.layers.normalization import BatchNormalization
    from sklearn.model_selection import train_test_split
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
    from keras.utils import to_categorical
    from wavhandler import transform_data

    top_weights_path = TEMP_DATADIR + str(model_name) + '.h5'
    targets = len(np.unique(y))

    if setting == 'psd_dB':
        X = transform_data(X, setting=setting)

    X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(X,y, random_state=0)

    # Convert label to onehot
    y_train = to_categorical(y_train, num_classes=targets)
    y_val = to_categorical(y_val, num_classes=targets)
    y_test = to_categorical(y_test, num_classes=targets)

    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Build the Neural Network
    model = Sequential()

    model.add(Conv1D(16, 3, activation='relu', input_shape=(X.shape[1], 1)))
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

    model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data = [X_val, y_val], callbacks = callbacks_list)

    model.load_weights(top_weights_path)
    loss, acc = model.evaluate(X_test, y_test, batch_size=16)

    print('loss', loss)
    print('Test accuracy:', acc)

    from keras.models import model_from_yaml
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(TEMP_DATADIR + model_name + "_raw_final.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    model.save_weights(TEMP_DATADIR + model_name + "_raw_weights_final.h5")

def make_classification_conv2d(X_names, y, model_name='test_', setting='stft', undersampling=True):
    seed = 2018
    np.random.seed(seed)
    import soundfile as sf
    from keras.applications.densenet import DenseNet121
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from keras.layers import Input
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

    current_model = DenseNet121

    # Training setting - What kind of data to use
    if setting in ['gasf','gadf', 'mtf', 'rp']:
        input_shape = (150,150,1)
    elif setting == 'stft':
        input_shape = (129, 120, 1)
    elif setting == 'melspec':
        input_shape = (128, 120, 1)
    else:
        raise ValueError('No valid data setting provided')

    # More settings
    model_name = model_name + '_' + setting + '_' + current_model.__name__
    top_weights_path = TEMP_DATADIR + str(model_name) + '.h5'
    logfile = TEMP_DATADIR + str(model_name) + '.log'
    batch_size = 32
    monitor = 'val_acc'
    es_patience = 7
    rlr_patience = 3
    target_names = np.unique(y)

    y = LabelEncoder().fit_transform(y)
    X = X_names#np.array(X_names).reshape(-1,1)

    X, y = shuffle(X, y, random_state=0) 

    if undersampling:
        from imblearn.under_sampling import RandomUnderSampler
        ros = RandomUnderSampler(random_state=0)
        ros.fit(X,y)
        X, y = ros.fit_resample(X,y)
        X = pd.Series(X.ravel()).tolist()
        print('After undersampling: \n{}\n'.format(pd.DataFrame(y).iloc[:,0].value_counts()))
    else:
        print('Class balance: \n{}\n'.format(pd.DataFrame(y).iloc[:,0].value_counts()))

    X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(X,y, random_state=0)

    # Model parameters
    img_input = Input(shape = input_shape)
    model = current_model(input_tensor = img_input, classes = len(target_names), weights = None)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    # Define Callbacks
    callbacks_list = [ModelCheckpoint(monitor = monitor,
                                    filepath = top_weights_path,
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
                        CSVLogger(filename = logfile)]
    # TRAIN
    model.fit_generator(train_generator(X_train,
                                        y_train, 
                                        batch_size=batch_size, 
                                        target_names=target_names,
                                        setting=setting),
                        steps_per_epoch = int(math.ceil(float(len(X_train)) / float(batch_size))),
                        epochs=100, 
                        validation_data = valid_generator(X_val,
                                                        y_val, 
                                                        batch_size=batch_size, 
                                                        target_names=target_names,
                                                        setting=setting), 
                        validation_steps = int(math.ceil(float(len(X_test)) / float(batch_size))),
                        callbacks = callbacks_list)
    # EVALUATE
    model.load_weights(top_weights_path)
    loss, acc = model.evaluate_generator(valid_generator(X_test, 
                                                        y_test, 
                                                        batch_size=batch_size, 
                                                        setting=setting, 
                                                        target_names=target_names),
            steps = int(math.ceil(float(len(X_test)) / float(batch_size))))

    print('loss', loss)
    print('Test accuracy:', acc)

    # SAVING
    from keras.models import model_from_yaml
    model_yaml = model.to_yaml()
    with open(TEMP_DATADIR + model_name + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    model.save_weights(TEMP_DATADIR + model_name + "_weights.h5")

def make_autoencoder_2d():####UNDER CONSTRUCTION####
    from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
    from keras.models import Model
    from keras.datasets import mnist
    from keras.callbacks import TensorBoard
    from keras import backend as K
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle

    input_img = Input(shape=(28, 28, 1))    # adapt this if using 'channels_first' image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8), i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # To train it, use the original MNIST digits with shape (samples, 3, 28, 28),
    # and just normalize pixel values between 0 and 1

    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))    # adapt this if using 'channels_first' image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))       # adapt this if using 'channels_first' image data format

    # open a terminal and start TensorBoard to read logs in the autoencoder subdirectory
    # tensorboard --logdir=autoencoder

    autoencoder.fit(x_train, x_train, epochs=50, batch_size=128, shuffle=True, validation_data=(x_test, x_test),
                    callbacks=[TensorBoard(log_dir='conv_autoencoder')], verbose=2)

    # take a look at the reconstructed digits
    decoded_imgs = autoencoder.predict(x_test)

def train_test_val_split(X,y, random_state=0, verbose=1):
    from sklearn.model_selection import train_test_split
    X, y = shuffle(X, y, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.10,
                                                    stratify=y, 
                                                    random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                    test_size=0.2, 
                                                    stratify=y_train,
                                                    random_state=0)
    if isinstance(X_train, np.ndarray):
        print("X_train shape: \t{}\nX_test shape:\t{}\nX_val shape:\t{}\n".format(X_train.shape, X_test.shape, X_val.shape))
    return X_train, X_test, X_val, y_train, y_test, y_val
