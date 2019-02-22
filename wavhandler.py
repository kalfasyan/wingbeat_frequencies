import numpy as np
import glob
from natsort import natsorted
import pandas as pd
import random
from utils import *
import logging

logger = logging.getLogger()
logger.setLevel(logging.WARN)

mosquitos_6 = ['Ae. aegypti', 'Ae. albopictus', 
                'An. gambiae', 'An. arabiensis', 
                'C. pipiens', 'C. quinquefasciatus']
increasing_dataset = ['aedes_male', 'fuit_flies', 'house_flies', 'new_aedes_female', 
                        'new_stigma_male','new_tarsalis_male', 'quinx_female', 'quinx_male', 
                        'stigma_female', 'tarsalis_female']
dros_zapr = ['LG_drosophila_10_09', 'LG_zapr_26_09']
dros_zapr_szki = ['LG_drosophila_10_09', 'LG_zapr_26_09', 'LG_suzukii_18_09_faulty']

class WavHandler(object):
    """ """
    def __init__(self, directory, sample_size=-1, recursive=False, nat_sort=True):
        """
        :param directory: the directory where the wav files are.
        :param sample_size: this refers to how many wav file to keep (randomly samples)
        """
        super(WavHandler, self).__init__()
        self.directory = directory
        self.sample_size = sample_size

        if recursive: # recursively selecting all files from directory and subdirectories within it.
            self.wav_filenames = list(glob.iglob(directory+'/**/*.wav', recursive=True))
        else:
            self.wav_filenames = glob.glob(directory+'*.wav')

        if nat_sort: # sorting file names naturally (natural sorting)
            self.wav_filenames = natsorted(self.wav_filenames)

        if isinstance(sample_size, int): # checking sample size and sampling
            if sample_size > -1:
                self.wav_filenames = random.sample(self.wav_filenames, sample_size)
        elif isinstance(sample_size, str):
            if sample_size == 'all':
                pass
        else:
            raise ValueError('Wrong sample_size given!')

        if len(self.wav_filenames) == 0: # raising error if nothing is returned.
            raise ValueError('No filenames retrieved!')

def get_data(filedir = '/home/kalfasyan/data/insects/Wingbeats/',
            target_names=mosquitos_6,
            nr_signals=20000,
            only_names=True,
            verbose=0):
    import os
    import soundfile as sf
    from tqdm import tqdm
    # Read about 'nr_signals' of recs from every species
    # Note: All wav files must be the same sampling frequency and number of datapoints!

    X = []                    # holds all data
    y = []                    # holds all class labels

    filenames = []            # holds all the file names
    target_count = []         # holds the counts in a class

    for i, target in enumerate(tqdm(target_names)):
        target_count.append(0)  # initialize target count
        path=filedir + target + '/'    # assemble path string

        for [root, dirs, files] in os.walk(path, topdown=False):
            for filename in files:
                name,ext = os.path.splitext(filename)
                if ext=='.wav':
                    name=os.path.join(root, filename)
                    if not only_names:
                        data, fs = sf.read(name)
                        X.append(data)
                    y.append(i)
                    filenames.append(name)
                    target_count[i]+=1
                    if target_count[i]>nr_signals:
                        break
        if verbose:
            print (target,'#recs = ', target_count[i])

    if not only_names:
        X = np.vstack(X)
        y = np.hstack(y)

        X = X.astype("float32")
        if verbose:
            print ("")
            print ("Total dataset size:")
            print ('# of classes: %d' % len(np.unique(y)))
            print ('total dataset size: %d' % X.shape[0])
            print ('Sampling frequency = %d Hz' % fs)
            print ("n_samples: %d" % X.shape[1])
            print ("duration (sec): %f" % (X.shape[1]/fs))

    return (X, y, filenames) if not only_names else (filenames, y)

def read_simple(paths, return_df=False):
    """
    Function to read wav files into a numpy array given their paths.
    It also returns their names for verification purposes.
    Return a dataframe if return_df is True.
    """
    import soundfile as sf
    data = []
    names = []
    for wav_ind, wavname in enumerate(paths):
        wavdata, fs = sf.read(wavname)
        #assert fs == 8000 and wavdata.shape[0] == 5000
        data.append(wavdata)
        names.append(wavname)
    datamatrix = np.asarray(data).T
    #assert datamatrix.shape == (5000, len(paths))
    logging.debug('datamatrix array created')
    logging.debug('names list created')

    if return_df:
        datamatrix = pd.DataFrame(datamatrix,
                      columns=[names[i].split('/')[-1][:-4] for i in range(datamatrix.shape[1])])
        return datamatrix, names
    else:
        return datamatrix, names
# Mel-scale filterbank features

def transform_data(X, setting = None):
    from scipy import signal
    from tqdm import tqdm
    # transform the data
    if setting=='spectrograms':
        import librosa
        XX = []#np.zeros((X.shape[0],129*120))
        for i in tqdm(range(X.shape[0])):
            data = librosa.stft(X[i], n_fft = N_FFT, hop_length = HOP_LEN)
            data = librosa.amplitude_to_db(np.abs(data))
            XX.append(np.flipud(data).flatten())
        XX = np.vstack(XX)
    elif setting=='melbank':
        XX = np.zeros((X.shape[0],80)).astype("float32")   # allocate space
        for i in range(X.shape[0]):
           XX[i] = np.log10(np.mean(librosa.feature.melspectrogram(X[i], sr=F_S, n_mels=80), axis=1))
    else: # default transformation is welch PSD 
        XX = np.zeros((X.shape[0],129)).astype("float32")   # allocate space
        for i in tqdm(range(X.shape[0])):
            XX[i] = 10*np.log10(signal.welch(X[i], fs=F_S, window='hanning', nperseg=256, noverlap=128+64)[1])
            # XX[i] = power_spectral_density(X[i], only_powers=True)
    return XX

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
    wavname = path
    fname = path.split('/')[-1][:-4]
    wavdata, fs = sf.read(path)
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
    return df

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

def get_wingbeat_timestamp(path):
    import pandas as pd
    fname = path.split('/')[-1]
    t = fname.split('_')[0] + fname.split('_')[1]
    return pd.to_datetime(t, format='F%y%m%d%H%M%S').hour