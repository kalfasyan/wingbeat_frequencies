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
        self.setting = 'raw'

    def read(self, data='all',fext='wav', labels='text', loadmat=True, setting='raw'):
        """
        Function to read wingbeat data with possibility to expand for image data as well.
        """
        import time
        self.setting = setting
        tic = time.time()
        assert isinstance(data, str) or isinstance(data, int), "Unsupported format for data. Give str or int."
        all_data = list(glob.iglob(self.directory + '/**/*.{}'.format(fext), recursive=True))

        # Reading Filenames of data
        # Load ALL data of dataset
        if data=='all': 
            self.filenames = all_data
        # Load only specified class
        elif data in self.target_classes: 
            tmpfnames = pd.Series(glob.iglob(self.directory + '/**/*.{}'.format(fext), recursive=True))
            basedirlen = len(self.base_dir.split('/'))
            self.filenames = tmpfnames[tmpfnames.apply(lambda x: x.split('/')[basedirlen]) == data]
        # In case an integer is provided
        elif data < 0:
            raise ValueError("Provide a positive integer for number of data")
        # Load part of data 
        else:
            self.filenames = all_data
            # If given number is smaller than total
            if data < len(self.filenames):
                self.filenames = random.sample(self.filenames, data)
            # If given number is larger than total
            else:
                print("Provided larger number than total nr of signals. Reading all data available")
                self.filenames = all_data
        assert len(self.filenames), "No data found."
        self.filenames = pd.Series(self.filenames)
        print("Data: {}.\nRead {} filenames in {:.2f} seconds.".format(data, len(self.filenames) ,time.time() - tic))

        # Reading values of data
        if loadmat:
            # If raw data is needed
            if setting == 'raw':
                self.X = pd.DataFrame(read_simple(self.filenames)[0])
            # Load data according to setting provided
            else:
                self.X = make_df_parallel(names=self.filenames, setting=setting)
            print("Loaded data into matrix in {:.2f} seconds.".format(time.time()-tic))

        # Reading labels
        if labels=='text':
            self.y = pd.Series(self.filenames).apply(lambda x: x.split('/')[x.split('/').index(self.name)+1])
        elif labels=='nr':
            from sklearn.preprocessing import LabelEncoder
            self.y = pd.Series(self.filenames).apply(lambda x: x.split('/')[x.split('/').index(self.name)+1])                
            le = LabelEncoder()
            self.y = pd.Series(le.fit_transform(self.y))
        else:
            raise ValueError('Wrong value given for labels argument.')

    def clean(self, threshold=10, plot=True):
        assert self.setting == 'psd_dB', "Cleaning works with psd_dB setting"

        self.y.index = list(self.y.reset_index(drop=True).index)
        self.X['var'] = self.X.apply(lambda x: x.iloc[10:50].var(), axis=1)
        inds = self.X[(self.X['var'] > threshold)].index

        if plot:
            np_hist(self.X, 'var')

        self.X, self.y = self.X.loc[inds].drop('var',axis=1).dropna(), self.y.loc[inds].dropna()
        self.filenames = self.filenames.loc[inds]

    def select_class(self, selection=None, fext='wav'):
        if not selection is None:
            assert isinstance(selection, str), "Provide a string of the class you want to select."
            assert selection in self.target_classes, "Selection given not found in dataset classes: {}".format(self.target_classes)
            self.target_classes = selection
            self.name = "{}/{}".format(self.name, selection)
            self.nr_classes = 1
            inds = self.y[self.y == selection].index
            self.filenames = self.filenames[inds]
            self.X = self.X.iloc[inds,:]
            self.y = self.y[inds]
        else:
            raise ValueError("Wrong selection given.")        

    def get_sensor_features(self, version='1',temp_humd=True, hist_temp=False, hist_humd=False, hist_date=False):
        assert hasattr(self, 'filenames')
        df = pd.DataFrame(self.filenames, columns=['filenames'])
        df['wavnames'] = df['filenames'].apply(lambda x: x.split('/')[-1][:-4])
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
                plt.figure(figsize=(10,6))
                df.date.hist(xrot=45)
                plt.ylabel('Counts of signals')
            self.df_features = df
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
            self.df_features = df

    def get_frequency_peaks(self, filter_signal=False):
        from scipy.signal import find_peaks

        assert hasattr(self, 'X'), 'Load the data first.'
        assert isinstance(self.X, np.ndarray) or isinstance(self.X, pd.DataFrame) 
        assert self.setting == 'raw', "You need to read the raw data to get frequency peaks."

        X = self.X.values if isinstance(self.X, pd.DataFrame) else self.X
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
        
    def plot_activity_times(self):
        import matplotlib.pyplot as plt
        df = pd.DataFrame(self.filenames.apply(lambda x: get_wingbeat_timestamp(x)).value_counts())
        df['counts'] = df[0]
        df['ind'] = df.index
        df.counts.sort_index().plot(kind='bar', figsize=(14,10))
        plt.ylabel('Counts of signals')
        plt.xlabel('Hour of the day (24H)')
        plt.title('Activity times of {}'.format(self.name))
        plt.show()
    
    def split(self, random=True):
        if random:
            from utils_train import train_test_val_split
            self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val = train_test_val_split(self.filenames, self.y, 
                                                                                                                random_state=0, 
                                                                                                                verbose=1, 
                                                                                                                test_size=0.10, 
                                                                                                                val_size=0.2)
        else:
            pass


def read_simple(paths):
    """
    Function to read wav files into a numpy array given their paths.
    It also returns their names for verification purposes.
    """
    import soundfile as sf
    data = []
    names = []
    for _, wavname in enumerate(paths):
        wavdata, _ = sf.read(wavname)
        data.append(wavdata)
        names.append(wavname)
    datamatrix = np.asarray(data)
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
    fname = path.split('/')[-1][:-4] # Filename is the last list element of full path without extension (.wav)
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
    elif setting == 'raw':
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

    # TODO_IDEA: make this pass a dictionary only with a string.format for the 0/1/2
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
