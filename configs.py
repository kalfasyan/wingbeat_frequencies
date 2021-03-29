__all__ = ['ModelConfiguration', 'TrainConfiguration', 'DatasetConfiguration']

import datetime
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
# MODEL CONFIG
from tensorflow.keras.applications.densenet import (DenseNet121, DenseNet169,
                                                    DenseNet201)
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.nasnet import NASNetLarge, NASNetMobile
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping,
                                        ModelCheckpoint, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.keras.layers import (GRU, LSTM, Activation, Add,
                                     BatchNormalization, Conv1D, Conv2D, Dense,
                                     Dropout, GlobalAveragePooling1D,
                                     GlobalAveragePooling2D, Input,
                                     MaxPooling1D, MaxPooling2D, add)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

from utils import TEMP_DATADIR
from wavhandler import BASE_DIR

seed = 42
np.random.seed(seed)

class DatasetConfiguration(object):
    def __init__(self, names=['Wingbeats','Thomas','Pcfruit','Pcfruit_sensor49','LG','Leafminers', 'Rodrigo','Suzukii_RL']):
        assert os.path.isdir(BASE_DIR), "Check BASE_DIR"
        assert len(names)
        self.base_dir = BASE_DIR
        self.names = names
        paths = [f"{os.path.join(BASE_DIR,i)}" for i in names]
        for i in paths:
            print(f"Dataset - {i.split('/')[-1]} - exists: {os.path.isdir(i)}")
        assert all([os.path.isdir(i) for i in paths]), "Non-existent dataset given."
        self.paths = paths
        self.species = []
        self.species_paths = []
        self.selected = False

    def info(self):
        print([{self.names[i]:os.listdir(self.paths[i])} for i in range(len(self.paths))])

    def select(self, name=None, species='all'):
        assert name in self.names,  "Unknown dataset selection."
        dataset_path = os.path.join(BASE_DIR,name)

        if species == 'all':
            species = os.listdir(dataset_path)

        assert all([os.path.isdir(f"{dataset_path}/{s}") for s in species])
        self.species.extend([f"{name}/{s}" for s in species])
        self.species_paths.extend([f"{dataset_path}/{s}/" for s in species])
        self.selected = True

        self.species = pd.Series(self.species).unique().tolist()
        self.species_paths = pd.Series(self.species_paths).unique().tolist()

    def select_all(self):
        for name in self.names:
            self.select(name=name, species='all')

    def read(self):
        import glob
        assert len(self.species) and len(self.species_paths)

        filenames = {s.split('/')[-1]:[] for s in self.species}
        for s,p in zip(self.species, self.species_paths):
            s = s.split('/')[-1]
            filenames[s].extend(list(glob.iglob(f"{p}/**/*.{'wav'}", recursive=True)))

        self.fnames_dict = filenames
        self.fnames = pd.Series(sum(self.fnames_dict.values(),[]))
        self.labels = pd.Series(self.fnames).apply(lambda x: x.split('/')[len(BASE_DIR.split('/'))])
        self.target_classes = [s.split('/')[-1] for s in self.species]
        self.df = pd.concat([self.fnames, self.labels], axis=1)
        self.df.columns = ['fnames','labels']

    def clean(self, low_threshold=8.9, high_threshold=20):
        from datahandling import get_clean_wingbeats_multiple_runs

        scores = get_clean_wingbeats_multiple_runs(names=self.fnames.tolist())
        self.df['score'] = scores
        self.df = self.df[(self.df['score'] > low_threshold) & (self.df['score'] < high_threshold)]
        self.fnames = self.df.fnames
        self.labels = self.df.labels

    def parse_filenames(self, version='1',temp_humd=True, hist_temp=False, hist_humd=False, hist_date=False):
        """
        Since the stored fnames contain metadata, this function gets all these features and 
        constructs a pandas Dataframe with them.
        """
        from utils import np_hist
        assert hasattr(self, 'fnames')
        df = self.df
        df['wavnames'] = df['fnames'].apply(lambda x: x.split('/')[-1][:-4])
        # LightGuide sensor version
        if version=='1':                        
            df['date'] = df['wavnames'].apply(lambda x: pd.to_datetime(''.join(x.split('_')[0:2]), 
                                                                        format='F%y%m%d%H%M%S'))
            df['datestr'] = df['date'].apply(lambda x: x.strftime("%Y%m%d"))
            df['date_day'] = df['date'].apply(lambda x: x.day)
            df['date_hour'] = df['date'].apply(lambda x: x.hour)
            df['gain'] = df['wavnames'].apply(lambda x: x.split('_')[3:][1])
            if temp_humd:
                df['temperature'] = pd.to_numeric(df['wavnames'].apply(lambda x: x.split('_')[3:][3] if len(x.split('_')[3:])>=3 else np.nan))
                df['humidity'] = pd.to_numeric(df['wavnames'].apply(lambda x: x.split('_')[3:][5] if len(x.split('_')[3:])>=4 else np.nan))
            if hist_temp:
                np_hist(df, 'temperature')
            if hist_humd:
                np_hist(df, 'humidity')
            if hist_date:
                import matplotlib.pyplot as plt
                df.datestr.sort_values().value_counts()[df.datestr.sort_values().unique()].plot(kind='bar', figsize=(22,10))
                plt.ylabel('Counts of signals')
            self.df_info = df
        # Fresnel sensor version
        elif version=='2':
            print('VERSION 2')
            df['date'] = df['wavnames'].apply(lambda x: pd.to_datetime(''.join(x.split('_')[1]), 
                                                                        format='%Y%m%d%H%M%S'))
            df['datestr'] = df['date'].apply(lambda x: x.strftime("%Y%m%d"))
            df['date_day'] = df['date'].apply(lambda x: x.day)
            df['date_hour'] = df['date'].apply(lambda x: x.hour)
            df['index'] = df['wavnames'].apply(lambda x: x.split('_')[2])
            if temp_humd:
                df['temperature'] = pd.to_numeric(df['wavnames'].apply(lambda x: x.split('_')[3][4:]))
                df['humidity'] = pd.to_numeric(df['wavnames'].apply(lambda x: x.split('_')[4][3:]))
            if hist_temp:
                np_hist(df, 'temperature')
            if hist_humd:
                np_hist(df, 'humidity')
            if hist_date:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10,6))
                df.date.hist(xrot=45)
                plt.ylabel('Counts of signals')
            self.df_info = df
        else:
            print("No sensor features collected. Select valid version")
        self.sensor_features = True

    def plot_daterange(self, start='', end='', figx=8, figy=26, linewidth=4):
        """
        Method to plot a histogram within a date range (starting from earliest datapoint to latest)
        """
        assert hasattr(self, 'sensor_features'), "Parse filenames first to generate features."

        import matplotlib.pyplot as plt

        from utils import get_datestr_range

        if '' in {start, end}:
            start = self.df_info.datestr.sort_values().iloc[0]
            end = self.df_info.datestr.sort_values().iloc[-1] 
        all_dates = get_datestr_range(start=start,end=end)

        hist_dict = self.df_info.datestr.value_counts().to_dict()
        mydict = {}
        for d in all_dates:
            if d not in list(hist_dict.keys()):
                mydict[d] = 0
            else:
                mydict[d] = hist_dict[d]

        series = pd.Series(mydict)
        ax = series.sort_index().plot(xticks=range(0,series.shape[0]), figsize=(figy,figx), rot=90, linewidth=linewidth)
        ax.set_xticklabels(series.index);


class ModelConfiguration(object):

    def __init__(self, model_setting=None, data_setting=None, nb_classes=None, extra_dense_layer=False):

        super(ModelConfiguration, self).__init__()

        self.model_setting = model_setting
        self.nb_classes = nb_classes

        # PRE-TRAINED MODELS
        self.supported_models = ['DenseNet121','DenseNet169','DenseNet201',
                        'InceptionResNetV2','VGG16','VGG19']

        if model_setting == 'DenseNet121':
            current_model = DenseNet121
        elif model_setting == 'DenseNet169':
            current_model = DenseNet169
        elif model_setting == 'DenseNet201':
            current_model = DenseNet201
        elif model_setting == 'InceptionResNetV2':
            current_model = InceptionResNetV2
        elif model_setting == 'VGG16':
            current_model = VGG16
        elif model_setting == 'VGG19':
            current_model = VGG19

        if data_setting in ['stft', 'stftflt']:
            self.input_shape = (129, 120, 1)
        elif data_setting in ['raw', 'rawflt']:
            self.input_shape = (5000, 1)
            if model_setting.endswith('baseline'):
                self.input_shape = (5000,1,1)
        elif data_setting in ['psd_dB', 'psd_dBflt']:
            self.input_shape = (129, 1)
        elif data_setting in ['cwt', 'cwtflt']:
            self.input_shape = (127, 127, 1)
        elif data_setting in ['psd', 'psdflt']:
            self.input_shape = (4097,1)
        else:
            raise ValueError('Wrong data_setting provided.')

        print(f'############ INPUT SHAPE:{self.input_shape}')

        if model_setting in self.supported_models:
            model = current_model(input_tensor = Input(shape = self.input_shape), 
                                classes = self.nb_classes, 
                                weights = None)
        # CUSTOM MODELS
        elif model_setting in ['gru','lstm']:
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
            model.add(Dense(self.nb_classes, activation=None))
            model.add(Activation('softmax'))
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
            model.add(Dense(self.nb_classes, activation=None))
            model.add(Activation('softmax'))
        elif model_setting == 'conv1d':
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
            model.add(Dense(self.nb_classes, activation=None))
            model.add(Activation('softmax'))
        elif model_setting == 'conv1d2':
            model = Sequential()
            model.add(Conv1D(16, 3, activation='relu', input_shape=self.input_shape))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(2))

            model.add(Conv1D(32, 3, activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(2))
            
            model.add(Conv1D(64, 3, activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(2))
            
            model.add(Conv1D(128, 3, activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(2))

            model.add(Conv1D(256, 3, activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(2))
            model.add(Dropout(0.5))

            model.add(GlobalAveragePooling1D())

            model.add(Dense(256, activation='relu'))
            model.add(Dense(self.nb_classes, activation=None))
            model.add(Activation('softmax'))

        # OTHER MODELS
        elif model_setting == 'dl4tsc_inc':
            NB_EPOCHS = 100
            filters = 32
            depth = 7
            kernel_size = 14
            print(TEMP_DATADIR)
            clf = Classifier_INCEPTION(TEMP_DATADIR, self.input_shape, self.nb_classes,
                                    nb_filters=filters, use_residual=True,
                                    use_bottleneck=True, depth=depth,
                                    kernel_size=int(kernel_size), verbose=False,
                                    batch_size=64, nb_epochs=NB_EPOCHS)
            model = clf.build_model(input_shape=self.input_shape, nb_classes=self.nb_classes)

        elif model_setting == 'conv1d_baseline':
            x = Input(shape=(self.input_shape))
        #    drop_out = Dropout(0.2)(x)
            conv1 = Conv2D(16, 3, 1, padding='same')(x)
            conv1 = BatchNormalization()(conv1)
            conv1 = Activation('relu')(conv1)
        #    drop_out = Dropout(0.2)(conv1)
            conv2 = Conv2D(32, 3, 1, padding='same')(conv1)
            conv2 = BatchNormalization()(conv2)
            conv2 = Activation('relu')(conv2)
        #    drop_out = Dropout(0.2)(conv2)
            conv3 = Conv2D(128, 3, 1, padding='same')(conv2)
            conv3 = BatchNormalization()(conv3)
            conv3 = Activation('relu')(conv3)
            # drop_out = Dropout(0.2)(conv3)
            conv4 = Conv2D(256, 3, 1, padding='same')(conv3)
            conv4 = BatchNormalization()(conv4)
            conv4 = Activation('relu')(conv4)
            full = GlobalAveragePooling2D()(conv4)
            out = Dense(self.nb_classes, activation='softmax')(full)
            model = Model(inputs=x, outputs=out)

        elif model_setting == 'dl4tsc_fcn':
            """
            Credits to: https://github.com/hfawaz/dl-4-tschttps://github.com/hfawaz/dl-4-tsc
            """
            model = Sequential()
            model.add(Conv1D(128, 8, padding='same', input_shape=self.input_shape))
            model.add(BatchNormalization())
            model.add(Activation(activation='relu'))
            model.add(Conv1D(256, 5, padding='same'))
            model.add(BatchNormalization())
            model.add(Activation(activation='relu'))
            model.add(Conv1D(128, 3, padding='same'))
            model.add(BatchNormalization())
            model.add(Activation(activation='relu'))
            model.add(GlobalAveragePooling1D())
            model.add(Dense(self.nb_classes, activation=None))
            model.add(Activation('softmax'))

        elif model_setting == 'tsc_fcn_baseline':
            x = Input(shape=(self.input_shape))
        #    drop_out = Dropout(0.2)(x)
            conv1 = Conv2D(128, 8, 1, padding='same')(x)
            conv1 = BatchNormalization()(conv1)
            conv1 = Activation('relu')(conv1)
        #    drop_out = Dropout(0.2)(conv1)
            conv2 = Conv2D(256, 5, 1, padding='same')(conv1)
            conv2 = BatchNormalization()(conv2)
            conv2 = Activation('relu')(conv2)
        #    drop_out = Dropout(0.2)(conv2)
            conv3 = Conv2D(128, 3, 1, padding='same')(conv2)
            conv3 = BatchNormalization()(conv3)
            conv3 = Activation('relu')(conv3)
            full = GlobalAveragePooling2D()(conv3)
            out = Dense(self.nb_classes, activation='softmax')(full)
            model = Model(inputs=x, outputs=out)

        elif model_setting == 'tsc_res_baseline':
            """ 
            Credits to: https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline
            """
            n_feature_maps = 64
            input_shape = self.input_shape
            x = Input(shape=(input_shape))
            conv_x = BatchNormalization()(x)
            conv_x = Conv2D(n_feature_maps, 8, 1, padding='same')(conv_x)
            conv_x = BatchNormalization()(conv_x)
            conv_x = Activation('relu')(conv_x)
            conv_y = Conv2D(n_feature_maps, 5, 1, padding='same')(conv_x)
            conv_y = BatchNormalization()(conv_y)
            conv_y = Activation('relu')(conv_y)
            conv_z = Conv2D(n_feature_maps, 3, 1, padding='same')(conv_y)
            conv_z = BatchNormalization()(conv_z)
            is_expand_channels = not (input_shape[-1] == n_feature_maps)
            if is_expand_channels:
                shortcut_y = Conv2D(n_feature_maps, 1, 1,padding='same')(x)
                shortcut_y = BatchNormalization()(shortcut_y)
            else:
                shortcut_y = BatchNormalization()(x)
            y = Add()([shortcut_y, conv_z])
            y = Activation('relu')(y)
            x1 = y
            conv_x = Conv2D(n_feature_maps*2, 8, 1, padding='same')(x1)
            conv_x = BatchNormalization()(conv_x)
            conv_x = Activation('relu')(conv_x)
            conv_y = Conv2D(n_feature_maps*2, 5, 1, padding='same')(conv_x)
            conv_y = BatchNormalization()(conv_y)
            conv_y = Activation('relu')(conv_y)
            conv_z = Conv2D(n_feature_maps*2, 3, 1, padding='same')(conv_y)
            conv_z = BatchNormalization()(conv_z)
            is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
            if is_expand_channels:
                shortcut_y = Conv2D(n_feature_maps*2, 1, 1,padding='same')(x1)
                shortcut_y = BatchNormalization()(shortcut_y)
            else:
                shortcut_y = BatchNormalization()(x1)
            y = Add()([shortcut_y, conv_z])
            y = Activation('relu')(y)
            x1 = y
            conv_x = Conv2D(n_feature_maps*2, 8, 1, padding='same')(x1)
            conv_x = BatchNormalization()(conv_x)
            conv_x = Activation('relu')(conv_x)
            conv_y = Conv2D(n_feature_maps*2, 5, 1, padding='same')(conv_x)
            conv_y = BatchNormalization()(conv_y)
            conv_y = Activation('relu')(conv_y)
            conv_z = Conv2D(n_feature_maps*2, 3, 1, padding='same')(conv_y)
            conv_z = BatchNormalization()(conv_z)
            is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
            if is_expand_channels:
                shortcut_y = Conv2D(n_feature_maps*2, 1, 1,padding='same')(x1)
                shortcut_y = BatchNormalization()(shortcut_y)
            else:
                shortcut_y = BatchNormalization()(x1)
            y = Add()([shortcut_y, conv_z])
            y = Activation('relu')(y)
            
            full = GlobalAveragePooling2D()(y)
            out = Dense(self.nb_classes, activation='softmax')(full)
            model = Model(inputs=x, outputs=out)

        elif model_setting == 'dl4tsc_res':
            n_feature_maps = 64
            input_layer = Input(self.input_shape)
            # BLOCK 1 
            conv_x = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
            conv_x = BatchNormalization()(conv_x)
            conv_x = Activation('relu')(conv_x)
            conv_y = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
            conv_y = BatchNormalization()(conv_y)
            conv_y = Activation('relu')(conv_y)
            conv_z = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
            conv_z = BatchNormalization()(conv_z)
            # expand channels for the sum 
            shortcut_y = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
            shortcut_y = BatchNormalization()(shortcut_y)
            output_block_1 = add([shortcut_y, conv_z])
            output_block_1 = Activation('relu')(output_block_1)
            # BLOCK 2 
            conv_x = Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1)
            conv_x = BatchNormalization()(conv_x)
            conv_x = Activation('relu')(conv_x)
            conv_y = Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
            conv_y = BatchNormalization()(conv_y)
            conv_y = Activation('relu')(conv_y)
            conv_z = Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
            conv_z = BatchNormalization()(conv_z)
            # expand channels for the sum 
            shortcut_y = Conv1D(filters=n_feature_maps*2, kernel_size=1, padding='same')(output_block_1)
            shortcut_y = BatchNormalization()(shortcut_y)
            output_block_2 = add([shortcut_y, conv_z])
            output_block_2 = Activation('relu')(output_block_2)
            # BLOCK 3 
            conv_x = Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_2)
            conv_x = BatchNormalization()(conv_x)
            conv_x = Activation('relu')(conv_x)
            conv_y = Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
            conv_y = BatchNormalization()(conv_y)
            conv_y = Activation('relu')(conv_y)
            conv_z = Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
            conv_z = BatchNormalization()(conv_z)
            # no need to expand channels because they are equal 
            shortcut_y = BatchNormalization()(output_block_2)
            output_block_3 = add([shortcut_y, conv_z])
            output_block_3 = Activation('relu')(output_block_3)
            # FINAL 
            gap_layer = GlobalAveragePooling1D()(output_block_3)
            output_layer = Dense(self.nb_classes, activation='softmax')(gap_layer)
            model = Model(inputs=input_layer, outputs=output_layer)
        else:
            raise ValueError("Wrong model setting given.")

        if extra_dense_layer:
            # Creating a new model to add a penultimate Dense layer
            # tmp_model is the model without its last softmax
            tmp_model = Model(model.inputs, model.layers[-2].output)
            # adding a Dense layer
            x = Dense(self.nb_classes, activation='relu')(tmp_model.layers[-1].output)
            x = Dense(self.nb_classes, activation='softmax')(x)

            del model
            model = Model(inputs=tmp_model.inputs, outputs=x)

        self.config = model

class TrainConfiguration(object):
    """ Configuration for training procedures. Contains all settings """
    def __init__(self, nb_classes=None, setting='raw', model_name='test', batch_size=32, monitor='val_loss', 
                es_patience=7, rlr_patience=3, epochs=100):

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
        self.nb_classes = nb_classes
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

import time


# resnet model
class Classifier_INCEPTION:
    from tensorflow import keras

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, batch_size=64,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500):

        self.output_directory = output_directory

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)
        gap_layer2 = keras.layers.Dense(nb_classes, activation=None)(gap_layer)

        output_layer = keras.layers.Activation('softmax')(gap_layer2)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model
