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

	def read(self, create_table=False):
		"""
		Reads the wavhandler files ínto a pandas dataframe
		If create_table is True, then it creates a data table for the wavhandler object.
		This table contains information about the files and the corresponding signals.
		"""
		datamatrix, names = read_simple(self.wav_filenames, return_df=True)

		if create_table:
			df = pd.DataFrame(names, columns=['names'])
			df['class'] = df.names.apply(lambda x: x.split('/')[-4])
			df['subclass'] = df.names.apply(lambda x: x.split('/')[-3])
			df['date'] = df.names.apply(lambda x: pd.to_datetime(x.split('/')[-2],format='D_%y_%m_%d_%H_%M_%S'))
			df['path_len'] = df.names.apply(lambda x: len(x.split('/')))
			df['fname'] = df.names.apply(lambda x: x.split('/')[-1][:-4])
			df['fname_len'] = df.names.apply(lambda x: len(x.split('/')[-1][:-4].split('_')))
			df['humidity'] = df.fname.apply(lambda x: x.split('_')[-1] if len(x.split('_')) == 9 else np.nan)
			df['temperature'] = df.fname.apply(lambda x: x.split('_')[-3] if len(x.split('_')) == 9 else np.nan)
			self.df_table = df
			logging.debug('df_table created')
		else:
			self.df_signals = pd.DataFrame(datamatrix, columns=[names[i].split('/')[-1][:-4] for i in range(datamatrix.shape[1])])
			logging.debug('df_signals created')

	def preprocess(self):
		"""
		Crops the signals using a rolling mean with a specified window as the number of values 
		in the abs(signal) to calculate the mean for. If window=500, then the moving average uses 
		500 values from the signal, then finds the max of this new signal and extracts the range 
		to use in order to crop the original signal.
		"""
		if not len(self.df_signals):
			raise ValueError('No df_signals. Read the wav files first!')

		df_processed = pd.DataFrame()
		for col in self.df_signals:
			sig = self.df_signals[col]
			sig = butter_bandpass_filter(data=sig, lowcut=L_CUTOFF, highcut=H_CUTOFF, fs=F_S, order=B_ORDER)
			sig = crop_signal(sig, window=300, intens_threshold=0.0004, offset=200)			
			df_processed[col] = sig
		self.df_signals = df_processed
		logging.debug('df_signals processed')

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

def get_psd(fname, data, plot=False):

    sig_bandpass = butter_bandpass_filter(data=data, lowcut=L_CUTOFF, highcut=H_CUTOFF, fs=F_S, order=B_ORDER)
    sig_cropped = crop_signal(sig_bandpass, window=300, intens_threshold=0.0004, offset=200)

    if sig_cropped is None or sig_cropped.empty:
        return pd.Series(np.ones(2500,)*np.nan)

    psd = psd_process(sig_cropped, fs=F_S, scaling='density', window='hamming', nfft=8192, noverlap=None, crop_hz=2500)
    psd[fname] = psd.pow_amp
    return psd[fname]

def process_signal(fname, data, plot=False):

	specs = {}
	results = {
#		'intens0': np.nan, 
#		'intens1': np.nan, 
#		'intens2': np.nan,
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

	sig_bandpass = butter_bandpass_filter(data=data, lowcut=L_CUTOFF, highcut=H_CUTOFF, fs=F_S, order=B_ORDER)
	sig_cropped = crop_signal(sig_bandpass, window=300, intens_threshold=0.0004, offset=200)

	if sig_cropped is None or sig_cropped.empty:
		specs[fname] = results
		return specs

#	sig_top_intens = pd.Series(np.abs(sig_cropped)).nlargest(3).tolist()
#	results['intens0'], results['intens1'], results['intens2'] = sig_top_intens[0], sig_top_intens[1], sig_top_intens[2]

	psd = psd_process(sig_cropped, fs=F_S, scaling='density', window='hamming', nfft=8192, noverlap=None, crop_hz=2500)
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
		plt.subplot(3,1,1); plt.plot(data); plt.title('raw')
		plt.subplot(3,1,2); plt.plot(sig_bandpass); plt.title('bandpassed')
		plt.subplot(3,1,3); plt.plot(psd.frequency, psd.pow_amp); plt.title('psd')

	specs[fname] = results
	return specs
	#[x]TODO: define/find through function/whatever ...the fundamental and harmonics...as well as
	#		the top 3 PSD amplitudes.
	#[x] ALSO: crop PSD signal up to 2500
	#[ ] ALSO: Check how to correct the userwarning in nperseg = 156 is greater than input length (perhaps save the input length)
	#[ ] LATER: PCA on covariance matrix, not correlation. Set 'with_std'=False in StandardScaler