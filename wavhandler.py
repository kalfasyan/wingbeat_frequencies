import numpy as np
np.random.seed(42)
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
        self.cleaned = False
        self.sensor_features = False
        self.class_path_idx = self.directory.split('/').index(self.name) + 1

    def read(self, data='all',fext='wav', labels='text', loadmat=True, setting='raw'):
        """
        Function to read wingbeat data with possibility to expand for image data as well.
        Choice of whether to get text labels or encoded numerical values.
        If loatmat is True, then it will load data, depending on the setting provided, into a matrix.
        """
        import time
        self.setting = setting
        tic = time.time()
        assert isinstance(data, str) or isinstance(data, int), "Unsupported format for data. Give str or int."
        all_data = list(glob.iglob(self.directory + '/**/*.{}'.format(fext), recursive=True))

        ## Reading Filenames of data
        # In case a string is provided
        if isinstance(data, str):
            assert data in self.target_classes + ['all'], 'Wrong species given / not understood'
            if data=='all': 
                self.filenames = all_data
            # Load only specified class
            else:
                tmpfnames = pd.Series(glob.iglob(self.directory + '/**/*.{}'.format(fext), recursive=True))
                basedirlen = len(self.base_dir.split('/'))
                self.filenames = tmpfnames[tmpfnames.apply(lambda x: x.split('/')[basedirlen]) == data]
                
        # In case an integer is provided
        elif isinstance(data, int):
            if data < 0:
                raise ValueError("Provide a positive integer for number of data")
            # If given number is smaller than total
            if data < len(self.filenames):
                self.filenames = random.sample(self.filenames, data)
            # If given number is larger than total
            else:
                print("Provided larger number than total nr of signals. Reading all data available")
                self.filenames = all_data
        assert len(self.filenames), "No data found."
        self.filenames = pd.Series(self.filenames)
        print("Species: {}.\nRead {} filenames in {:.2f} seconds.".format(data, len(self.filenames) ,time.time() - tic))

        ## Reading data into a matrix
        if loadmat:
            self.X = self.make_array(setting=setting)
            print("Loaded data into matrix in {:.2f} seconds.".format(time.time()-tic))

        ## Reading labels
        if labels=='text':
            self.y = pd.Series(self.filenames).apply(lambda x: x.split('/')[x.split('/').index(self.name)+1])
        elif labels=='nr':
            from sklearn.preprocessing import LabelEncoder
            self.y = pd.Series(self.filenames).apply(lambda x: x.split('/')[x.split('/').index(self.name)+1])                
            le = LabelEncoder()
            self.y = pd.Series(le.fit_transform(self.y))
        else:
            raise ValueError('Wrong value given for labels argument.')

    def clean(self, threshold=10, threshold_interf=0, plot=False):
        """
        Cleans the dataset depending on the variance of their spectrum.
        """
        # TODO: Make this function read with multiprocessing the psd_dB, then return cleaned filenames etc.
        # just like remove pests
        assert self.setting == 'psd_dB', "Cleaning works with psd_dB setting"

        self.filenames.reset_index(drop=True, inplace=True)
        self.y.index = list(self.y.reset_index(drop=True).index)
        self.X['var_fly_vs_walk'] = self.X.apply(lambda x: x.iloc[4:].var(), axis=1)
        self.X['var_interference'] = self.X.apply(lambda x: x.iloc[:5].var(), axis=1)
        inds = self.X[(self.X['var_fly_vs_walk'] > threshold) & (self.X['var_interference'] > threshold_interf)].index

        if plot:
            np_hist(self.X, 'var')

        self.X, self.y = self.X.loc[inds].drop(['var_fly_vs_walk', 
                                                'var_interference'],axis=1).dropna(), self.y.loc[inds].dropna()
        self.filenames = self.filenames.loc[inds]
        self.cleaned = True
        print("{} filenames after cleaning.".format(len(self.filenames)))

    def remove_f0_range(self, low=95, high=105):
        """
        Given a range from low to high, calculate the PSD of all signals 
        and remove those that have a peak withing this range.
        """
        from scipy import signal as sg
        from scipy.signal import find_peaks

        assert self.cleaned, "Needs to be cleaned first."
        if not hasattr(self, 'psd'):
            self.psd = make_df_parallel(names=self.filenames.tolist(), setting='psd')
        inds = []
        for i in range(len(self.psd)):
            peaks, _ = find_peaks(self.psd.iloc[i].values)
            if not len(np.where((peaks > low) & (peaks < high))[0]):
                inds.append(i)
        inds = np.array(inds)
        self.filenames, self.X, self.y = self.filenames.iloc[inds], self.X.iloc[inds], self.y.iloc[inds]
        print("{} filenames after removing interference.".format(len(self.filenames)))

    def make_array(self, setting=None):
        """
        Makes use of make_df_parallel which creates a pandas Dataframe using the multiprocessing 
        library for parallel computation. Depending on the setting, a different Dataframe is created.
        """
        assert hasattr(self, 'filenames'), "Read data first"
        if setting == 'raw':
            self.raw = make_df_parallel(names=self.filenames.tolist(), setting=setting)
            return self.raw
        elif setting == 'psd':
            self.psd = make_df_parallel(names=self.filenames.tolist(), setting=setting)
            return self.psd
        elif setting == 'psd_dB':
            self.psd_dB = make_df_parallel(names=self.filenames.tolist(), setting=setting)
            return self.psd_dB
        elif setting == 'stft':
            self.specs = make_df_parallel(names=self.filenames.tolist(), setting=setting)
            return self.specs
        else:
            raise ValueError('No such setting exists / is implemented')


    def get_night_signals(self, after=21, before=8):
        """
        Select signal waveform in a specific time interval i.e. 'before' some time and 'after' some time.
        """
        assert self.setting == 'psd_dB', "This works with psd_dB setting only"
        assert self.cleaned == True, "This works with cleaned datasets only"
        assert self.sensor_features == True, "Retrieve sensor features first"

        sub = self.df_features
        inds = sub[(sub.date_hour >= after) | (sub.date_hour <= before)].index.values

        self.filenames = self.filenames.loc[inds]
        self.X = self.X.loc[inds]
        self.y = self.y.loc[inds]
        print("{} night signal filenames.".format(len(self.filenames)))

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
        """
        Since the stored filenames contain metadata, this function gets all these features and 
        constructs a pandas Dataframe with them.
        """
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
        else:
            print("No sensor features collected. Select valid version")
        self.sensor_features = True

    def get_frequency_peaks(self, filter_signal=False, lcut=L_CUTOFF, hcut=H_CUTOFF, fs=F_S):
        """
        Creates a histogram plot with counts of dominant frequencies.
        """
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
        df = df[df['freqs'] < 450]
        np_hist(df,'freqs')
        
    def plot_activity_times(self):
        """
        Plots the activity times of all data in the Dataset. Useful when selecting specific species.
        """
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
            self.X_train, self.X_test, \
                self.X_val, self.y_train, \
                    self.y_test, self.y_val = train_test_val_split(self.filenames, 
                                                                    self.y, 
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
    if setting=='stft':
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

def transform_data_parallel_psd(path):
    from scipy import signal as sg
    x, _ = read_simple([path])
    f,p = sg.welch(x.ravel(), fs=8000, scaling='density', window='hanning', nfft=8128, nperseg=256, noverlap=128+64)
    p = pd.Series(p[:1000])
    p.index = f[:1000]
    return p

def make_df_parallel(setting=None, names=None):
    import multiprocessing
    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpus)
    result_list = []
    if setting == 'psd_old':
        result_list.append(pool.map(power_spectral_density_parallel, names))
    elif setting == 'psd':
        result_list.append(pool.map(transform_data_parallel_psd, names))
    elif setting == 'raw':
        result_list.append(pool.map(read_simple_parallel, names))
    elif setting == 'stft':
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
    return pd.to_datetime(t, format='F%y%m%d%H%M%S')
