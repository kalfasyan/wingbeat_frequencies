import numpy as np
import glob
from natsort import natsorted
import pandas as pd
import random
from utils import *
import logging


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

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

all_6 = ['Ae. aegypti', 'Ae. albopictus', 'An. gambiae', 'An. arabiensis', 'C. pipiens', 'C. quinquefasciatus']
mosquitos_6 = ['Ae. aegypti', 'Ae. albopictus', 'An. gambiae', 'An. arabiensis', 'C. pipiens', 'C. quinquefasciatus']
dros_zapr = ['LG_drosophila_10_09', 'LG_zapr_26_09']

def get_data(target_names=all_6, nr_signals=20000, only_names=True, verbose=0):
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
        path='/home/kalfasyan/data/insects/Wingbeats/' + target + '/'    # assemble path string

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

    return (X, y) if not only_names else (filenames, y)

def transform_data(X):
    from scipy import signal
    from tqdm import tqdm
    # transform the data
    XX = np.zeros((X.shape[0],129)).astype("float32")   # allocate space
    for i in tqdm(range(X.shape[0])):
        XX[i] = 10*np.log10(signal.welch(X[i], fs=F_S, window='hanning', nperseg=256, noverlap=128+64)[1])
        # XX[i] = power_spectral_density(X[i], only_powers=True)
    return XX

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
        assert fs == 8000 and wavdata.shape[0] == 5000
        data.append(wavdata)
        names.append(wavname)
    datamatrix = np.asarray(data).T
    assert datamatrix.shape == (5000, len(paths))
    logging.debug('datamatrix array created')
    logging.debug('names list created')

    if return_df:
        datamatrix = pd.DataFrame(datamatrix,
                      columns=[names[i].split('/')[-1][:-4] for i in range(datamatrix.shape[1])])
        return datamatrix, names
    else:
        return datamatrix, names

def read_simple_parallel(path):
    import soundfile as sf
    wavname = path
    fname = path.split('/')[-1][:-4]
    wavdata, fs = sf.read(path)
    wavseries = pd.Series(wavdata)
    wavseries.name = fname
    return wavseries

def power_spectral_density(data=None, fname=None, only_powers=False,crop=False):
    # Buttersworth bandpass filter
    sig = butter_bandpass_filter(data=data.flatten(), lowcut=L_CUTOFF, highcut=H_CUTOFF, fs=F_S, order=B_ORDER)
    if crop: # Perform cropping
        sig = crop_signal(sig, window=300, intens_threshold=0.0004, offset=200)
        if sig is None or sig.empty or sig.shape[0] < 256:
            logging.warning('Signal is None, empty or too small after cropping!')
            return None

    psd = psd_process(sig, fs=F_S, scaling='density', window='hanning', nperseg=256, noverlap=128+64)#, crop_hz=2500)
    return psd.pow_amp if only_powers else psd

def power_spectral_density_parallel(path):
    data, _ = read_simple([path])
    fname = path.split('/')[-1][:-4]
    psd_pow_amps = power_spectral_density(data=data.flatten(), fname=fname, only_powers=True, crop=False)
    psd_pow_amps.name = fname
    return psd_pow_amps

def make_df_parallel(df, setting=None, names=None):
    import multiprocessing
    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpus)
    result_list = []
    if setting == 'psd':
        result_list.append(pool.map(power_spectral_density_parallel, names))
    elif setting == 'read':
        result_list.append(pool.map(read_simple_parallel, names))
    else:
        logging.error('Wrong setting!')
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
