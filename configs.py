__all__ = ['ModelConfiguration', 'TrainConfiguration']

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization,Input, LSTM, GRU, MaxPooling2D
from tensorflow.keras.layers import Conv1D, Conv2D, Add,GlobalAveragePooling1D, MaxPooling1D, GlobalAveragePooling2D
from tensorflow.keras.layers import add
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
# MODEL CONFIG
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.nasnet import NASNetLarge, NASNetMobile
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception

from utils import TEMP_DATADIR
import datetime
import numpy as np
seed = 42
np.random.seed(seed)

class ModelConfiguration(object):
    def __init__(self, model_setting=None, data_setting=None, target_names=None):

        super(ModelConfiguration, self).__init__()

        self.model_setting = model_setting
        self.target_names = target_names
        self.nb_classes = len(target_names)

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

        if data_setting == 'stft':
            self.input_shape = (129, 120, 1)
        elif data_setting == 'raw':
            self.input_shape = (5000, 1)
            if model_setting.endswith('baseline'):
                self.input_shape = (5000,1,1)
        elif data_setting == 'psd_dB':
            self.input_shape = (129, 1)
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
            model.add(Dense(self.nb_classes, activation='softmax'))
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
            model.add(Dense(self.nb_classes, activation='softmax'))
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
            model.add(Dense(self.nb_classes, activation='softmax'))

        # OTHER MODELS
        elif model_setting == 'dl4tsc_inc':
            NB_EPOCHS = 100
            filters = 32
            depth = 7
            kernel_size = 14
            clf = Classifier_INCEPTION(TEMP_DATADIR, self.input_shape, self.nb_classes,
                                    nb_filters=filters, use_residual=True,
                                    use_bottleneck=True, depth=depth,
                                    kernel_size=int(kernel_size), verbose=False,
                                    batch_size=64, nb_epochs=NB_EPOCHS)
            model = clf.build_model(input_shape=self.input_shape, nb_classes=self.nb_classes)

        elif model_setting == 'conv1d_baseline':
            x = Input(shape=(self.input_shape))
            conv1 = Conv2D(16, 3, 1, padding='same', activation='relu')(x)
            conv1 = Conv2D(16, 3, 1, padding='same', activation='relu')(conv1)
            conv1 = BatchNormalization()(conv1)
            conv2 = Conv2D(32, 3, 1, padding='same', activation='relu')(conv1)
            conv2 = Conv2D(32, 3, 1, padding='same', activation='relu')(conv2)
            conv2 = BatchNormalization()(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 1), strides=None, padding='same')(conv2)
            conv3 = Conv2D(64, 3, 1, padding='same', activation='relu')(pool2)
            conv3 = Conv2D(64, 3, 1, padding='same', activation='relu')(conv3)
            conv3 = BatchNormalization()(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 1), strides=None, padding='same')(conv3)
            conv4 = Conv2D(128, 3, 1, padding='same', activation='relu')(pool3)
            conv4 = Conv2D(128, 3, 1, padding='same', activation='relu')(conv4)
            conv4 = BatchNormalization()(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 1), strides=None, padding='same')(conv4)
            conv5 = Conv2D(256, 3, 1, padding='same', activation='relu')(pool4)
            conv5 = Conv2D(256, 3, 1, padding='same', activation='relu')(conv5)
            conv5 = BatchNormalization()(conv5)
            full = GlobalAveragePooling2D()(conv5)
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
            model.add(Dense(self.nb_classes, activation='softmax'))

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

        self.config = model

class TrainConfiguration(object):
    """ Configuration for training procedures. Contains all settings """
    def __init__(self, dataset=None, setting='raw', model_name='test', batch_size=32, monitor='val_loss', 
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

# resnet model
from tensorflow import keras
import numpy as np
import time


class Classifier_INCEPTION:

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

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

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