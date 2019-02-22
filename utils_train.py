from utils import *
from wavhandler import *
from keras.utils import np_utils
from keras.preprocessing import image
import librosa
from scipy import signal
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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

def train_generator(X_train, y_train, batch_size, target_names, crop=False):
    while True:
        for start in range(0, len(X_train), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(X_train))
            train_batch = X_train[start:end]
            labels_batch = y_train[start:end]

            for i in range(len(train_batch)):
                data, rate = librosa.load(train_batch[i], sr = SR)
                if crop:
                    data = crop_rec(data)

                data = random_data_shift(data, u = .2)

                data = librosa.stft(data, n_fft = N_FFT, hop_length = HOP_LEN)
                data = librosa.amplitude_to_db(np.abs(data))
                data = np.flipud(data)

                data = np.expand_dims(data, axis = -1)
                data = random_data_shift(data, w_limit = (-0.25, 0.25), h_limit = (-0.0, 0.0), cval = np.min(data), u = 1.0)

                # data = np.squeeze(data, axis = -1)
                # plt.imshow(data, cmap = 'gray')
                # plt.show()
                # data = np.expand_dims(data, axis = -1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)

            y_batch = np_utils.to_categorical(y_batch, len(target_names))

            yield x_batch, y_batch

def valid_generator(X_test, y_test, batch_size, target_names, crop=False):
    while True:
        for start in range(0, len(X_test), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(X_test))
            test_batch = X_test[start:end]
            labels_batch = y_test[start:end]

            for i in range(len(test_batch)):
                data, rate = librosa.load(test_batch[i], sr = SR)
                if crop:
                    data = crop_rec(data)

                data = librosa.stft(data, n_fft = N_FFT, hop_length = HOP_LEN)
                data = librosa.amplitude_to_db(np.abs(data))
                data = np.flipud(data)

                data = np.expand_dims(data, axis = -1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)

            y_batch = np_utils.to_categorical(y_batch, len(target_names))

            yield x_batch, y_batch

def print_confusion(y_test, y_pred, target_names):
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, columns=target_names, index=target_names)
    plt.figure(figsize = (15,10))
    sns.heatmap(df_cm, annot=True, fmt="d")

    print("")

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, columns=target_names, index=target_names)
    plt.figure(figsize = (15,10))
    sns.heatmap(df_cm, annot=True)
    plt.show()


def crop_rec(data):

    f, t, Zxx = signal.stft(data, SR, nperseg=256)
    Z = np.sum(np.abs(Zxx), axis=0)
    max_pos = np.argmax(Z)
    mid_x = 1+128*max_pos
    nsamples = 5000
    mid_x = np.max([nsamples/2, mid_x])
    mid_x = np.min([len(data)-nsamples/2, mid_x])
    x = data[(-nsamples/2 + mid_x + range(nsamples)).astype(int)]

    # data original signal, x: cropped signal
    return x