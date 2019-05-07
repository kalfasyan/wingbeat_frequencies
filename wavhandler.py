import numpy as np
import glob, os
from natsort import natsorted
import pandas as pd
import random
from utils import *
import logging

logger = logging.getLogger()
logger.setLevel(logging.WARN)

BASE_DIR = '/home/kalfasyan/data/insects/'

class Dataset(object):
    """ """
    def __init__(self, name):
        super(Dataset, self).__init__()
        assert os.path.isdir(BASE_DIR)
        self.base_dir = BASE_DIR
        self.name = name
        self.directory = os.path.join(self.base_dir, self.name) + '/'
        assert os.path.isdir(self.directory), 'No such dataset found in {}. Check your folder structure.'.format(self.base_dir)
        self.target_classes = natsorted(os.listdir(os.path.join(self.base_dir, self.name)))
        self.nr_classes = len(self.target_classes)
        self.X = pd.DataFrame()
        self.y = []

    def load(self, nr_signals=np.inf, only_names=False, text_labels=False, verbose=0):
        import os
        import soundfile as sf
        from tqdm import tqdm
        from PIL import Image

        # dataset = self.name
        filedir = self.directory
        target_names= self.target_classes

        X = []                    # holds all data
        y = []                    # holds all class labels
        t = []                    # holds all class labels (text)

        filenames = []            # holds all the file names
        target_count = []         # holds the counts in a class

        if verbose:
            print('Gathering approximately {} signals from each class'.format(nr_signals))

        for i, target in enumerate(tqdm(target_names, disable=DISABLE_TQDM)):
            target_count.append(0)  # initialize target count
            path=filedir + target + '/'    # assemble path string

            for [root, dirs, files] in os.walk(path, topdown=False):
                for filename in files:
                    name,ext = os.path.splitext(filename)
                    if ext=='.wav' or ext=='.jpg':
                        name=os.path.join(root, filename)
                        if not only_names:
                            if ext=='.wav':
                                data, fs = sf.read(name)
                                if self.name == 'increasing dataset':
                                    data = crop_rec(data)
                            elif ext=='.jpg':
                                temp = Image.open(name)
                                data = np.array(temp.copy())
                                temp.close()
                            X.append(data)
                        y.append(i)
                        t.append(target)
                        filenames.append(name)
                        target_count[i]+=1
                        if target_count[i]>nr_signals:
                            break
            if verbose:
                print (target,'#recs = ', target_count[i])
        if text_labels:
            y = t
        if not only_names:
            if not self.name.startswith('MOSQUITOES_IMGS'):
                X = np.vstack(X)
                X = X.astype("float32")
            y = np.hstack(y)

            if verbose:
                print ("")
                print ("Total dataset size:")
                print ('# of classes: %d' % len(np.unique(y)))
                print ('total dataset size: %d' % X.shape[0])
                print ('Sampling frequency = %d Hz' % fs)
                print ("n_samples: %d" % X.shape[1])
                print ("duration (sec): %f" % (X.shape[1]/fs))
        self.X = X
        self.y = y
        self.filenames = filenames

        return (X, filenames, y) if not only_names else (filenames, y)
    
    def get_sensor_features(self, temp_humd=True):
        assert hasattr(self, 'filenames')
        df = pd.DataFrame(self.filenames, columns=['filenames'])
        df['wavnames'] = df['filenames'].apply(lambda x: x.split('/')[-1][:-4])
        df['date'] = df['wavnames'].apply(lambda x: pd.to_datetime(''.join(x.split('_')[0:2]), 
                                                                    format='F%y%m%d%H%M%S'))
        df['date_day'] = df['date'].apply(lambda x: x.day)
        df['date_hour'] = df['date'].apply(lambda x: x.hour)
        df['gain'] = df['wavnames'].apply(lambda x: x.split('_')[3:][1])
        if temp_humd:
            df['temperature'] = pd.to_numeric(df['wavnames'].apply(lambda x: x.split('_')[3:][3]))
            df['humidity'] = pd.to_numeric(df['wavnames'].apply(lambda x: x.split('_')[3:][5]))
        self.df_features = df
        return df

    def get_frequency_peaks(self, filter_signal=False):
        from scipy.signal import find_peaks
        assert hasattr(self, 'X'), 'Load the data first.'
        assert isinstance(self.X, np.ndarray) or isinstance(self.X, pd.DataFrame) 
        if isinstance(self.X, pd.DataFrame):
            X = self.X.values
        freq_range = np.linspace(0, F_S/2, 129)
        freqs = []
        for i in range(X.shape[0]):
            sig = X[i,:]
            if filter_signal:
                sig = butter_bandpass_filter(sig, L_CUTOFF, H_CUTOFF, F_S, B_ORDER)
            sig_tr = transform_data(sig)
            peaks, _ = find_peaks(sig_tr)
            freqs = freqs + freq_range[peaks].tolist()
        df = pd.DataFrame(pd.to_numeric(pd.Series(freqs)), columns=['freqs'])
        df = df[df['freqs'] < 500]
        np_hist(df,'freqs')
    
    def read(self, nr_data='all',fext='wav', labels='text', loadmat=True, setting='read'):
        import glob

        if isinstance(nr_data, str) and nr_data=='all':
            self.filenames = list(glob.iglob(self.directory + '/**/*.{}'.format(fext), recursive=True))
        else:
            assert isinstance(nr_data, int) and nr_data > 0, "Provide a positive integer for number of data"
            self.filenames = list(glob.iglob(self.directory + '/**/*.{}'.format(fext), recursive=True))
            if nr_data < len(self.filenames):
                self.filenames = random.sample(self.filenames, nr_data)
            else:
                print("Provided larger number than total nr of signals. Reading all data available")
                self.filenames = list(glob.iglob(self.directory + '/**/*.{}'.format(fext), recursive=True))
        assert len(self.filenames), "No data found."

        if loadmat:
            # self.X = read_simple(self.filenames)[0]
            self.X = make_df_parallel(names=self.filenames, setting=setting)

        if labels=='text':
            self.y = pd.Series(self.filenames).apply(lambda x: x.split('/')[x.split('/').index(self.name)+1])
        elif labels=='nr':
            from sklearn.preprocessing import LabelEncoder
            self.y = pd.Series(self.filenames).apply(lambda x: x.split('/')[x.split('/').index(self.name)+1])                
            le = LabelEncoder()
            self.y = pd.Series(le.fit_transform(self.y))
        else:
            raise ValueError('Wrong value given for labels argument.')

def read_simple(paths, return_df=False):
    """
    Function to read wav files into a numpy array given their paths.
    It also returns their names for verification purposes.
    Return a dataframe if return_df is True.
    """
    import soundfile as sf
    data = []
    names = []
    for _, wavname in enumerate(paths):
        wavdata, _ = sf.read(wavname)
        data.append(wavdata)
        names.append(wavname)
    datamatrix = np.asarray(data)
    logging.debug('datamatrix array created')
    logging.debug('names list created')

    if return_df:
        datamatrix = pd.DataFrame(datamatrix,
                      columns=[names[i].split('/')[-1][:-4] for i in range(datamatrix.shape[1])])
        return datamatrix, names
    else:
        return datamatrix, names

def transform_data(X, setting = None):
    from scipy import signal
    from tqdm import tqdm
    import librosa
    # transform the data
    if setting=='spectrograms':
        XX = []#np.zeros((X.shape[0],129*120))
        for i in tqdm(range(X.shape[0]), disable=DISABLE_TQDM):
            data = librosa.stft(X[i], n_fft = N_FFT, hop_length = HOP_LEN)
            data = librosa.amplitude_to_db(np.abs(data))
            XX.append(np.flipud(data).flatten())
        XX = np.vstack(XX)
    elif setting=='melbank':
        XX = np.zeros((X.shape[0],80)).astype("float32")   # allocate space
        for i in range(X.shape[0]):
           XX[i] = np.log10(np.mean(librosa.feature.melspectrogram(X[i], sr=F_S, n_mels=80), axis=1))
    else: # default transformation is welch PSD 
        if len(X.shape) == 1:
            X = X.reshape(1,-1)
        XX = np.zeros((X.shape[0],129)).astype("float32")   # allocate space
        for i in tqdm(range(X.shape[0]), disable=DISABLE_TQDM):
            XX[i] = 10*np.log10(signal.welch(X[i], fs=F_S, window='hanning', nperseg=256, noverlap=128+64)[1])
            # XX[i] = power_spectral_density(X[i], only_powers=True)
    return XX.squeeze()

def power_spectral_density(data=None, fname=None, only_powers=False, crop=False, bandpass=False,
                            fs=F_S, scaling='density', window='hanning', nperseg=256, noverlap=128+64, nfft=None):
    from scipy import signal as sg
    from scipy.signal import find_peaks
    from sklearn.preprocessing import normalize

    if bandpass: # Buttersworth bandpass filter
        data = butter_bandpass_filter(data=data.flatten(), lowcut=L_CUTOFF, highcut=H_CUTOFF, fs=F_S, order=B_ORDER)
    if crop: # Perform cropping
        data = crop_signal(data, window=300, intens_threshold=0.0004, offset=200)
        if data is None or data.empty or data.shape[0] < 256:
            logging.warning('Signal is None, empty or too small after cropping!')
            return None
    # Calculating PSD
    freqs, p_amps = sg.welch(data, fs=fs, scaling=scaling, window=window, nfft=nfft, nperseg=nperseg, noverlap=noverlap)
    # Normalization of PSD amplitudes
    p_amps = normalize(p_amps.reshape(-1,1), norm='l2', axis=0).reshape(-1,)
    psd = pd.concat([pd.Series(freqs), pd.Series(p_amps)], axis=1)
    # Cropping up to 2500 Hz (mosquitos don't have more)
    # psd = psd.iloc[:crop_hz,:]
    psd.columns = ['frequency','pow_amp']

    return psd.pow_amp if only_powers else psd


def read_simple_parallel(path):
    import soundfile as sf
    fname = path.split('/')[-1][:-4]
    wavdata, _ = sf.read(path)
    wavseries = pd.Series(wavdata)
    wavseries.name = fname
    return wavseries

def transform_data_parallel(path):
    from scipy import signal
    x, _ = read_simple([path])
    x = 10*np.log10(signal.welch(x.ravel(), fs=F_S, window='hanning', nperseg=256, noverlap=128+64)[1])
    return pd.Series(x)

def transform_data_parallel_melbank(path):
    import librosa
    x, _ = read_simple([path])
    x = np.log10(np.mean(librosa.feature.melspectrogram(x.ravel(), sr=F_S, n_mels=80), axis=1)) 
    return pd.Series(x)

def transform_data_parallel_spectograms(path):
    import librosa
    x, _ = read_simple([path])
    x = x.ravel()
    x = librosa.stft(x, n_fft = N_FFT, hop_length = HOP_LEN)
    x = librosa.amplitude_to_db(np.abs(x))
    x = np.flipud(x).flatten()
    return pd.Series(x)

def power_spectral_density_parallel(path):
    X, _ = read_simple([path])
    fname = path.split('/')[-1][:-4]
    psd_pow_amps = power_spectral_density(data=X.flatten(), fname=fname, only_powers=True, crop=False)
    psd_pow_amps.name = fname
    return psd_pow_amps

def make_df_parallel(setting=None, names=None):
    import multiprocessing
    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpus)
    result_list = []
    if setting == 'psd':
        result_list.append(pool.map(power_spectral_density_parallel, names))
    elif setting == 'read':
        result_list.append(pool.map(read_simple_parallel, names))
    elif setting == 'spectrograms':
        result_list.append(pool.map(transform_data_parallel_spectograms, names))
    elif setting == 'psd_dB':
        result_list.append(pool.map(transform_data_parallel, names))
    elif setting == 'melbank':
        result_list.append(pool.map(transform_data_parallel_melbank, names))
    else:
        logging.error('Wrong setting!')
    pool.close()
    df = pd.concat(result_list[0], axis=1, sort=False)
    return df.T

def process_signal(data=None, fname=None, plot=False):
    specs = {}
    results = {
        'pow0': np.nan,
        'pow1': np.nan,
        'pow2': np.nan,
        'fr0': np.nan,
        'fr1': np.nan,
        'fr2': np.nan,
        'damping_0': np.nan,
        'damping_1': np.nan,
        'damping_2': np.nan,
    }
    # Calculate the power spectral density
    psd = power_spectral_density(data=data, fname=None, only_powers=False,crop=False)
    if psd is None:
        specs[fname] = results
        return specs

    peaks = peak_finder(psd, min_freq=300.)
    results['pow0'], results['fr0'], peak0 = get_harmonic(psd, peaks, h=0)
    results['pow1'], results['fr1'], peak1 = get_harmonic(psd, peaks, h=1)
    results['pow2'], results['fr2'], peak2 = get_harmonic(psd, peaks, h=2)

    # TODO: make this pass a dictionary only with a string.format for the 0/1/2
    results['damping_0'] = damping_ratio(fund_freq=results['fr0'], fund_amp=results['pow0'], psd=psd, peak_idx=peak0)
    results['damping_1'] = damping_ratio(fund_freq=results['fr1'], fund_amp=results['pow1'], psd=psd, peak_idx=peak1)
    results['damping_2'] = damping_ratio(fund_freq=results['fr2'], fund_amp=results['pow2'], psd=psd, peak_idx=peak2)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(24,6))
        plt.subplot(2,1,1); plt.plot(data); plt.title('raw')
        plt.subplot(2,1,2); plt.plot(psd.frequency, psd.pow_amp); plt.title('psd')

    specs[fname] = results
    return specs

def plot_wingbeat(data=None, plot=False):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,10))
    plt.subplot(2,2,1)
    plt.title('raw')
    plt.plot(data)
    plt.subplot(2,2,2)
    plt.title('filtered')
    plt.plot(butter_bandpass_filter(data, lowcut=L_CUTOFF, highcut=H_CUTOFF, fs=F_S, order=B_ORDER))
    plt.subplot(2,2,3)
    plt.title('transformed psd')
    plt.plot(transform_data(data))
    plt.show()

def get_wingbeat_timestamp(path):
    import pandas as pd
    fname = path.split('/')[-1]
    t = fname.split('_')[0] + fname.split('_')[1]
    return pd.to_datetime(t, format='F%y%m%d%H%M%S').hour

def merge_datasets_to_dict(dst1, dst2, namedtuple=False):
    
    d1 = dst1.copy()
    d2 = dst2.copy()
    for key,_ in d1.items():
        if isinstance(d1[key], list):
            l = [d2[key]]
            zz = [item for sublist in l for item in sublist]
            d1.update({key: d1[key]+zz})
        if isinstance(d1[key], str) and d1[key] != d2[key]:
            d1.update({key: d1[key] + '&' + d2[key]})
        if isinstance(d1[key], np.ndarray):
            if key == 'X':
                d1.update({key: np.vstack((d1[key], d2[key]))})
            elif key == 'y':
                d1.update({key: np.vstack((d1[key].reshape(-1,1), 
                                           d2[key].reshape(-1,1))).ravel()
                          })
    d1.update({'nr_classes': len(d1['target_classes'])})
    if len(d1['target_classes']) != len(np.unique(d1['target_classes'])):
        logging.warn('There are class duplicates!')

    if namedtuple:
        from collections import namedtuple
        return namedtuple("MergedDataset", d1.keys())(*d1.values())
    else:
        return d1
