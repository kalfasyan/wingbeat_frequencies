import os
import warnings
from configparser import ConfigParser
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import scipy.signal as sg
import soundfile as sf
from scipy import signal
from sklearn import preprocessing
from tensorflow.python.keras.utils import Sequence

from utils import butter_bandpass_filter

cfg = ConfigParser()
# cfg.read(f'config.ini')

BASE_DATAPATH = Path("/home/kalfasyan/data/wingbeats/")
BASE_PROJECTPATH = Path("/home/kalfasyan/projects/wingbeat_frequencies/")
L_CUTOFF = 100 
H_CUTOFF = 2500
B_ORDER = 4
N_FFT = 256
SR = 8000
HOP_LEN = int(N_FFT/6)


class WBDataset(object):
    def __init__(self, dsname, clean=True, custom_label=[], sample=0, verbose=True):

        self.dsname = dsname
        self.clean = clean
        self.sample = sample
        self.custom_label = custom_label
        self.verbose = verbose

        if self.clean:
            self.files, self.labels, self.lbl2files, self.sums = collect_wingbeat_files(dsname, sample=self.sample, clean=self.clean, verbose=True)
        else:
            self.files, self.labels, self.lbl2files = collect_wingbeat_files(dsname, sample=self.sample, clean=self.clean, verbose=True)

        if len(custom_label) == 1:
            self.labels = [custom_label[0] for _ in self.labels]
            if self.verbose:
                print(f"Label(s) changed to {custom_label}")
        else:
            if self.verbose:
                print("No custom label applied.")

    def __getitem__(self, idx):
        return {'path': self.files[idx], 'label': self.labels[idx]}

    def __len__(self):
        return len(self.files)

class WBDataGenerator(Sequence):
    def __init__(self, x, y, filtered=True, setting='raw', conv1d2d=False, batch_size=32):
        self.x, self.y, self.filtered, self.setting, self.batch_size = x, y, filtered, setting, batch_size
        self.conv1d2d = conv1d2d

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # reading data
        x = [load_wbt(filename, filtered=self.filtered, setting=self.setting, conv1d2d=self.conv1d2d) for filename in batch_x] 
        y = [label for label in batch_y]
        return np.array(x), np.array(y)


def get_xy_from_datasets(datasets=[]):
    files, labels = [], []
    for i, ds in enumerate(datasets):
        assert isinstance(ds, WBDataset), f"Number {i} is not a Winbgbeats dataset. Got type: {type(ds)}"
        files.extend(ds.files)
        labels.extend(ds.labels)
    return files, labels

def collect_wingbeat_files(dsname, sample=0, clean=True, low_threshold=8.9, high_threshold=20, verbose=True):
    datadir = Path(BASE_DATAPATH/dsname)

    files = []
    for file in datadir.glob("**/*.wav"):
        if not file.is_file():  # Skip directories
            continue
        files.append(file)
    print(f"Found {len(files)} files.")

    if clean:
        fname = f"{BASE_PROJECTPATH}/temp_data/{dsname.replace('/','-').replace(' ', '')}_thL{low_threshold}_thH{high_threshold}_cleaned"
        if os.path.isfile(f"{fname}.csv"):
            print("Found saved cleaned data.")
            df = pd.read_csv(f"{fname}.csv")
        else:
            print(f"Running cleaning process with low threshold: {low_threshold} and high: {high_threshold}")
            scores = get_clean_wingbeats_multiple_runs(names=files)
            df = pd.DataFrame({'score': scores, 'fnames': files})
            df = df[(df['score'] > low_threshold) & (df['score'] < high_threshold)]
            df.to_csv(f"{fname}.csv")
        files = df.fnames.tolist()
        sums = df.score.tolist()
        print(f"Total number of files after cleaning: {len(files)}.")

    if sample > 0:
        sampled_inds = np.random.choice(range(len(files)), sample, replace=True)
        if sample > len(files):
            print("Asked to sample more than the number of files found.")
        files = pd.Series(files).loc[sampled_inds].tolist()

    labels = pd.Series(files).apply(lambda x: label_func(x)).tolist() #list(files.map(label_func))
    lbl2files = {l: [f for f in files if label_func(f) ==l] for l in list(set(labels))}

    if clean:
        return files, labels, lbl2files, sums 
    else: 
        return files,labels, lbl2files

def label_func(fname):
    dsname = str(fname).split('/')[len(BASE_DATAPATH.parts)]
    if (dsname.startswith(("Suzukii_RL", "Melanogaster_RL"))):
        if dsname == 'Suzukii_RL':
            return 'D. suzukii'
        elif dsname == 'Melanogaster_RL':
            return 'D. melanogaster'
    else:
        return str(fname).split('/')[len(BASE_DATAPATH.parts)+1]

def read_wbt(path, filtered=True):
    data, _ = sf.read(path)
    if filtered:
        data = butter_bandpass_filter(data, L_CUTOFF, H_CUTOFF, fs=SR, order=B_ORDER)
    return data

def load_wbt(path, filtered=True, setting='stft', conv1d2d=False):
    data = read_wbt(path, filtered=filtered)

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
        pass
    elif setting == 'psd_dB':
        data = 10*np.log10(signal.welch(data, fs=SR, window='hanning', nperseg=256, noverlap=128+64)[1])
    elif setting.startswith('psd'):
        _,data = signal.welch(data, fs=SR, scaling='density', window='hanning', nfft=8192, nperseg=256, noverlap=128+64)
        if setting == 'psdl1':
            data = preprocessing.normalize(data.reshape(1,-1), norm='l1').T.squeeze()
        elif setting == 'psdl2':
            data = preprocessing.normalize(data.reshape(1,-1), norm='l2').T.squeeze()
    else:
        raise ValueError("Wrong setting given.")
    if conv1d2d:
        return np.expand_dims(np.expand_dims(data, axis=-1), axis=-1)
    else:
        return np.expand_dims(data, axis=-1)

def normalized_psd_sum(sig):
    _,p = sg.welch(sig, fs=8000, scaling='density', window='hanning', nfft=8192, nperseg=256, noverlap=128+64)
    p = preprocessing.normalize(p.reshape(1,-1), norm='l2').T.squeeze()
    return p.sum()

def psd_multiple_runs_score(path):
    x = read_wbt(path)
    x = x.ravel()
    x = butter_bandpass_filter(x, L_CUTOFF, H_CUTOFF, fs=SR, order=B_ORDER)
    score1 = normalized_psd_sum(x[:2501])
    score2 = normalized_psd_sum(x[2500:])
    score3 = normalized_psd_sum(x[1250:3750])
    return min([score1, score2, score3])

def get_clean_wingbeats_multiple_runs(names=''):
    import multiprocessing
    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpus)
    result_list = []
    result_list.append(pool.map(psd_multiple_runs_score, names))
    pool.close()
    return pd.Series(result_list[0])
