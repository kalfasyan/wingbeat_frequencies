import numpy as np
import glob
from natsort import natsorted
import pandas as pd
import random
from utils import *
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

L_CUTOFF = 75.
H_CUTOFF = 1000.
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
			self.df_signals = butter_dataframe(self.df_signals, 
											   L_CUTOFF, 
											   H_CUTOFF, 
											   F_S, 
											   order=B_ORDER)
			logging.debug('Buttersworth bandpass filter applied, [{}, {}, {}, order={}]'.format(L_CUTOFF,
																								H_CUTOFF,
																								F_S,
																								B_ORDER))
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


def evaluate(paths, df=None, butter_band=True):
	"""
	Evaluation function for wav signals given their paths.
	"""
	from scipy import signal
	from scipy.signal import find_peaks
	from sklearn.preprocessing import normalize

	if df is None:
		datamatrix, names = read_simple(paths)
		df = pd.DataFrame(datamatrix, columns=[names[i].split('/')[-1][:-4] for i in range(datamatrix.shape[1])])
	elif df.empty:
		raise ValueError('Empty Dataframe. Cannot evaluate!')	

	if butter_band:
		df = butter_dataframe(df, 
							  L_CUTOFF, 
							  H_CUTOFF, 
							  F_S, 
							  order=B_ORDER)
	good_sigs = []
	for col in df:
		sig = df[col]

		freqs, pows = signal.welch(sig, F_S, scaling='density', window='hamming')

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

def psd_welch_freqs(paths):
	from scipy import signal
	from scipy.signal import find_peaks

	df, names = read_simple(paths, return_df=True)
	freqs_dict = {}
	for col in df:
		sig = df[col].values
		freqs, pows = signal.welch(sig, F_S, scaling='density', window='hamming')

		pows = normalize(pows.reshape(-1,1), norm='l2', axis=0).reshape(-1,)
		threshold = 0.1
		
		peaks, _ = find_peaks(pows, height=threshold, distance=10)
		peaks = [v for i,v in enumerate(peaks) if freqs[peaks][i] > 400]

		sub = pd.DataFrame(np.vstack((freqs[peaks], pows[peaks])).T, columns=['freqs','pows'])
		peakseries = sub['pows'].nlargest(10)


		freqs_dict[col] = freqs[peaks].tolist()
	return freqs_dict

def psd_welch_pows(paths):
	from scipy import signal
	from scipy.signal import find_peaks

	df, names = read_simple(paths, return_df=True)
	pows_dict = {}
	for col in df:
		sig = df[col].values
		freqs, pows = signal.welch(sig, F_S, scaling='density', window='hamming')

		pows_dict[col] = pows	
	
	return pows_dict	

def signal_amplitudes(paths, top_nr=3):

	df, names = read_simple(paths, return_df=True)

	top_ampls = {}
	for col in df:
		top_ampls[col] = np.abs(df[col]).nlargest(top_nr).tolist()
	return top_ampls

def damping_ratio(freqs, ampls, peaks):
	fund_ampl = ampls[peaks[0]]
	fund_freq = freqs[peaks[0]]

	peak_a, peak_b = peaks[0], peaks[0]

	while ampls[peak_a] > fund_ampl/2:
		peak_a+=1
	while ampls[peak_b] > fund_ampl/2:
		peak_b-=1

	omega_a, omega_b = freqs[peak_a], freqs[peak_b]
	damping = (omega_a - omega_b) / (2*fund_freq)
	return damping

def damping_ratios(paths):
	#// TODO: make this find freqs, pows, peaks and then damping ratio
	df, names = read_simple(paths, return_df=True)

	damps = {}
	for col in df:
		damps[col] = None
	return None
