import numpy as np
import glob
from natsort import natsorted
import pandas as pd
import random
from utils import *
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

L_CUTOFF = 100.
H_CUTOFF = 2500.
F_S = 8000.
B_ORDER = 4

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

	def preprocess(self, butter_bandpass=True):
		"""
		Preprocesses the signal.
		If butter_bandpass is True, then it applies a Buttersworth band-pass filter with global variable settings.
		"""
		if not len(self.df_signals):
			raise ValueError('No df_signals. Read the wav files first!')
		if butter_bandpass:
			self.df_signals = butter_dataframe(self.df_signals, L_CUTOFF, H_CUTOFF, F_S, order=B_ORDER)
			logging.debug('Buttersworth bandpass filter applied, [{}, {}, {}, order={}]'.format(L_CUTOFF, H_CUTOFF, F_S,order=B_ORDER))
			self.preprocessed = True # Flag to check at filter_accepted_signals if the signals have been preprocessed
		else:
			return None

	def filter_accepted_signals(self):
		"""
		Filters the signals according to the evaluation performed in evaluate function.
		"""
		from scipy import signal
		from scipy.signal import find_peaks

		if not len(self.df_signals):
			raise ValueError('No df_signals. Read the wav files first!')
		if not hasattr(self, 'preprocessed'):
			raise ValueError('You have not preprocessed the data')

		if isinstance(self.df_signals, pd.DataFrame) and not self.df_signals.empty:
			# accepted_signals will be a list of the signal filenames that passed the evaluation
			self.accepted_signals = evaluate(self.wav_filenames, self.df_signals)
			# we select the accepted signals for the df_signals dataframe
			self.df_signals = self.df_signals[self.accepted_signals]
			logging.debug('df_signals filtered')
			logging.debug('accepted_signals created')
		else:
			raise ValueError('Not a DataFrame or empty DataFrame!')

	def crop(self, window=500):
		"""
		Crops the signals using a rolling mean with a specified window as the number of values 
		in the abs(signal) to calculate the mean for. If window=500, then the moving average uses 
		500 values from the signal, then finds the max of this new signal and extracts the range 
		to use in order to crop the original signal.
		"""
		if not len(self.df_signals):
			raise ValueError('No df_signals. Read the wav files first!')

		new_df = pd.DataFrame()
		for col in self.df_signals:
			sigseries = self.df_signals[col]
			rolling_mean = np.abs(sigseries).rolling(window).mean()
			cropped_signal = sigseries.iloc[rolling_mean.idxmax()+1-window:rolling_mean.idxmax()+1]
			new_df[col] = cropped_signal.values
		self.df_cropped = new_df
		logging.debug('df_cropped created')


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

def preprocess_simple(df, bandpass=True, crop=True):

	if bandpass:
		df = butter_dataframe(df, L_CUTOFF, H_CUTOFF, F_S, order=B_ORDER)
	if crop:
		df = crop_df(df, window=300, intens_threshold=0.004, offset=250)
	return df

def evaluate(df):
	"""
	Evaluation function for wav signals given their paths.
	"""
	from scipy import signal
	from scipy.signal import find_peaks
	from sklearn.preprocessing import normalize

	good_sigs = []
	for col in df:
		sig = df[col]

		freqs, pows = signal.welch(sig, F_S, scaling='density', window='hamming', nfft=8192, noverlap=None)

		pows = normalize(pows.reshape(-1,1), norm='l2', axis=0).reshape(-1,)

		threshold = 0.1
		peaks, vals = find_peaks(pows, height=threshold, distance=10)
		peaks = [v for i,v in enumerate(peaks) if freqs[peaks][i] > 400]

		damping = damping_ratio(freqs, pows, peaks)

		sub = pd.DataFrame(np.vstack((freqs[peaks], pows[peaks])).T, columns=['freqs','pows'])
		peakseries = sub['pows'].nlargest(10)

		if peakseries.shape[0] == 2:
			condition = (peakseries.iloc[0] > threshold or peakseries.iloc[1] > threshold)
		elif peakseries.shape[0] > 2:
			condition = (peakseries.iloc[0] > threshold or peakseries.iloc[1] > threshold) and \
					(peakseries.loc[0] > peakseries.loc[2]) and \
					(peakseries.loc[1] > peakseries.loc[2])
		else:
			condition = False
		if condition:
			good_sigs.append(col)
	logging.debug('list with accepted signals created')
	return good_sigs

def process_signal(data):

	sig_bandpass = butter_bandpass_filter(data, lowcut, highcut, fs, order=4)
	sig_cropped = crop_signal(sig_bandpass, window=300, intens_threshold=0.004, offset=250)

	sig_top_intens = pd.Series(np.abs(sig_cropped)).nlargest(3).tolist()

	freqs, p_amps, peaks = psd_process(sig_cropped, peak_thd=0.05, peak_dist=10, min_freq=400)

	damping = damping_ratio(freqs, p_amps, peaks)

	return None
	#TODO: define/find through function/whatever ...the fundamental and harmonics...as well as
	#		the top 3 PSD amplitudes.
	# ALSO: crop PSD signal up to 2500
	# LATER: PCA on covariance matrix, not correlation