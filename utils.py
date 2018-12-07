from scipy.signal import butter, lfilter
import pandas as pd
import numpy as np


def butter_bandpass(lowcut, highcut, fs, order=4):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b,a = butter(order, [low, high], btype='band')
	return b,a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
	b,a = butter_bandpass(lowcut, highcut, fs, order = order)
	y = lfilter(b,a,data)
	return y

def butter_dataframe(df, lowcut, highcut, fs, order=4):
	for col in df:
		y = butter_bandpass_filter(df[col].values, lowcut, highcut, fs, order=4)
		df[col] = y
	return df

def crop_signal(data, window=300, intens_threshold=0.004, offset=250):
	import more_itertools as mit

	sig = df[col].values
	sigseries = pd.Series(sig)
	rolling_avg = np.abs(sigseries).rolling(window).mean()
	rolling_avg_thd = rolling_avg[rolling_avg > intens_threshold]
	if len(rolling_avg_thd):
		iterable = rolling_avg_thd.index.tolist()
		groups = [list(group) for group in mit.consecutive_groups(iterable)]
		# Sizes of the groups
		group_lens = pd.Series([len(ww[i]) for i in range(len(ww))])
		# Index of largest group
		lrgst_group_idx = group_lens.idxmax()
		# The first element of the largest group is where we start cropping
		crop_start = groups[lrgst_group_idx][0]
		# The last element of the largest group is where we stop cropping
		crop_end = groups[lrgst_group_idx][-1]

		sig_cropped = sigseries.iloc[  crop_start -offset : crop_end[-1] - offset]
		return sig_cropped
	else:
		return None

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

def psd_process(data, peak_thd=0.05, peak_dist=10, min_freq=400):
	sig = data
	# Calculating PSD
	freqs, p_amps = signal.welch(sig, F_S, scaling='density', window='hamming', nfft=8192, noverlap=None)
	# Normalization of PSD amplitudes
	p_amps = normalize(p_amps.reshape(-1,1), norm='l2', axis=0).reshape(-1,)

	peaks, vals = find_peaks(p_amps, height=peak_thd, distance=peak_dist)
	peaks = [v for i,v in enumerate(peaks) if freqs[peaks][i] > min_freq]

	return freqs, p_amps, peaks

def tsfresh_transform(df):
	all_subs = []
	for i,col in enumerate(df):
		col_dict = {'ts_signal': df[col].values,
					'time': range(0, df[col].shape[0]),
					'id': i}
		all_subs.append(pd.DataFrame(col_dict))

	return pd.concat(all_subs, axis=0)

def make_tsfresh_settings(samples=300, feature_importances='feature_importances_v1.csv'):
	import tsfresh
	"""Having extracted initially the feature importances from a RF classifier that was trained
		using a number of 'samples', we can run this function to extract a tsfresh dictionary, 
		which we can pass so that the feature extraction takes less time, since it will use only 
		the important features from the past training """
	df = pd.read_pickle('combined_{}.pkl'.format(samples))
	featlist = pd.read_csv(feature_importances)
	df = df[featlist.names.tolist()]
	kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(df)

	return kind_to_fc_parameters

def create_classifier(name):
	name = name.lower()
	if name == 'logreg':
		from sklearn.linear_model import LogisticRegression
		classifier = LogisticRegression(random_state = 0, multi_class='auto', solver='lbfgs', max_iter=2000)
	elif name == 'knn': 
		from sklearn.neighbors import KNeighborsClassifier
		classifier = KNeighborsClassifier(n_neighbors = 14, metric = 'minkowski', p = 2)
	elif name == 'rf':
		from sklearn.ensemble import RandomForestClassifier
		classifier = RandomForestClassifier(n_estimators = 400, criterion = 'gini', random_state = 0)#, class_weight='balanced')
	elif name == 'svm':
		from sklearn.svm import SVC
		classifier = SVC(kernel='linear', random_state = 0)
	elif name == 'sgd':
		from sklearn.linear_model import SGDClassifier
		classifier = SGDClassifier(max_iter=3000, n_jobs=-1, tol=1e-3, random_state=0)
	elif name == 'extratree':
		from sklearn.ensemble import ExtraTreesClassifier
		classifier = ExtraTreesClassifier(n_estimators=80, random_state=0)
	elif name == 'xgboost':
		from xgboost import XGBClassifier
		classifier = XGBClassifier(n_estimators=400, n_jobs=-1, random_state=0)
	elif name == 'adaboost':
		from sklearn.ensemble import AdaBoostClassifier
		from sklearn.tree import DecisionTreeClassifier
		classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,
										algorithm="SAMME.R", learning_rate=0.5)
	elif name == 'mlp':
		from sklearn.neural_network import MLPClassifier
		classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, max_iter=1000, activation='relu',
									hidden_layer_sizes=(30,))#, learning_rate='adaptive')
	elif name == 'nbayes':
		from sklearn.naive_bayes import GaussianNB
		classifier = GaussianNB()
	else:
		raise ValueError('Wrong Classifier name given!')
	return classifier

