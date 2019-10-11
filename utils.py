import pandas as pd
import numpy as np
from scipy import signal
import logging
import os

B_ORDER = 4
L_CUTOFF = 120.
H_CUTOFF = 3600.
F_S = 8000.
N_FFT = 256
SR = 8000
HOP_LEN = int(N_FFT/6)
DISABLE_TQDM = True
TEMP_DATADIR = os.path.join(os.getcwd(), 'temp_data/')

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    from scipy.signal import butter, lfilter

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='bandpass')
    y = lfilter(b, a, data)
    return y

def butter_dataframe(df, lowcut, highcut, fs, order=4):
    for col in df:
        y = butter_bandpass_filter(df[col].values, lowcut, highcut, fs, order=4)
        df[col] = y
    return df

def crop_signal(data, window=300, intens_threshold=0.0004, offset=250):
    import more_itertools as mit

    sig = data
    sigseries = pd.Series(sig.reshape(-1,)) # fixing peculiarities
    rolling_avg = np.abs(sigseries).rolling(window).mean() # rolling average
    rolling_avg_thd = rolling_avg[rolling_avg > intens_threshold] # values above threshold
    if len(rolling_avg_thd) > N_FFT:
        iterable = rolling_avg_thd.index.tolist()
        groups = [list(group) for group in mit.consecutive_groups(iterable)]
        # Sizes of the groups
        group_lens = pd.Series([len(groups[i]) for i in range(len(groups))])
        # Index of largest group
        lrgst_group_idx = group_lens.idxmax()
        # The first element of the largest group is where we start cropping
        crop_start = groups[lrgst_group_idx][0]
        # The last element of the largest group is where we stop cropping
        crop_end = groups[lrgst_group_idx][-1]

        sig_cropped = sigseries.iloc[  crop_start -offset : crop_end - offset]
        return sig_cropped
    else:
        logging.debug('No values above intensity threshold!')
        return None

def peak_finder(psd, min_freq=0., only_freqs=False, h_threshold=0.01, d_threshold=3):
    from scipy.signal import find_peaks
    # Finding peaks in the whole PSD
    peaks, vals = find_peaks(psd.pow_amp, height=h_threshold, distance=d_threshold)
    peaks = [v for i,v in enumerate(peaks) if psd.frequency.iloc[peaks].iloc[i] > min_freq]
    return peaks if not only_freqs else psd.frequency.iloc[peaks].tolist()

def get_harmonic(psd, peaks, h=1):
    if len(peaks):
        from scipy.signal import find_peaks
        # Setting the fundamental frequency and the power amplitude
        fund_fr = psd.frequency.iloc[peaks].iloc[0]
        fund_amp = psd.pow_amp.iloc[peaks].iloc[0]
        # Defining the range to search the harmonic in
        har_range_idx = [fund_fr*(h+1)-100, fund_fr*(h+1)+100]
        har_range = psd.frequency[(psd.frequency>har_range_idx[0]) & \
                                  (psd.frequency<har_range_idx[1])].index.tolist()
        # Finding the same range in power amplitude and doing peak detection on it
        har_powamp = psd.pow_amp.iloc[har_range]
        har_fr = psd.frequency.iloc[har_range]
        if len(har_powamp)<=1 and len(har_fr)<=1:
            return np.nan, np.nan, np.nan
        # Index for the peak
        peak = find_peaks(har_powamp, distance=10)
        if len(peak[0]):
            peak = peak[0][0]
        else:
            return np.nan, np.nan, np.nan
        # Given the index, return the amplitude and frequency
        harmonic_pow = har_powamp.iloc[peak]
        harmonic_fr = har_fr.iloc[peak]
        harmonic_peak = har_fr.index[0]+ peak

        return harmonic_pow, harmonic_fr, harmonic_peak
    else:
        return np.nan, np.nan, np.nan

def damping_ratio(fund_freq, fund_amp, psd, peak_idx):

    if fund_freq is np.nan or fund_amp is np.nan:
        return np.nan

    peak_a, peak_b = peak_idx, peak_idx

    while psd.pow_amp[peak_a] > fund_amp/2:
        peak_a+=1
        if peak_a >= 2499:
            break
        if peak_a-peak_idx > 50:
            peak_a = peak_idx
            break
    while psd.pow_amp[peak_b] > fund_amp/2:
        if peak_idx-peak_b > 50:
            peak_b = peak_idx
            break
        peak_b-=1

    omega_a, omega_b = psd.frequency.iloc[peak_a], psd.frequency.iloc[peak_b]
    damping = (omega_a - omega_b) / (2*fund_freq)
    return damping

def tsfresh_transform(df):
    all_subs = []
    for i,col in enumerate(df):
        sig = df[col].dropna()
        col_dict = {'ts_signal': sig.values,
                    'time': range(0, sig.shape[0]),
                    'id': i,
                    }
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

def get_classifier(name):
    name = name.lower()
    if name == 'logreg':
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0, multi_class='auto', solver='lbfgs', max_iter=2000)
    elif name == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 14, metric = 'minkowski', p = 2)
    elif name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 325, n_jobs=-1, criterion = 'gini', random_state = 0, class_weight='balanced')
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
        classifier = XGBClassifier(n_estimators=325, n_jobs=-1, random_state=0, class_weight='balanced')
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

def crop_rec(data):

    f, t, Zxx = signal.stft(data, SR, nperseg=256)
    Z = np.sum(np.abs(Zxx), axis=0)
    max_pos = np.argmax(Z)
    mid_x = 1+128*max_pos
    nsamples = 5000
    mid_x = np.max([nsamples/2, mid_x])
    mid_x = np.min([len(data)-nsamples/2, mid_x])
    x = data[(-nsamples/2 + mid_x + range(nsamples)).astype(int)]

    # data original signal, x: cropped signal
    return x

def print_confusion(y_test, y_pred, target_names):
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, columns=target_names, index=target_names)
    plt.figure(figsize = (15,10))
    sns.heatmap(df_cm, annot=True, fmt="d")

    print("")

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, columns=target_names, index=target_names)
    plt.figure(figsize = (15,10))
    sns.heatmap(df_cm, annot=True)
    plt.show()

def similarity_matrix(mat, fun, axis=0, plot=False, uptr=False):
    assert hasattr(fun, '__call__'), "fun is not a function"
    initial = np.zeros((mat.shape[axis], 
                        mat.shape[axis]))
    for i, vec1 in enumerate(mat):
        for j, vec2 in enumerate(mat):
            if fun.__name__ == 'spearmanr':
                initial[i,j] = fun(vec1, vec2).correlation
            else:
                initial[i,j] = fun(vec1, vec2)	

    if uptr:
        return up_triang(initial)
    if plot:
        import seaborn as sns
        sns.heatmap(initial)
    return initial

def similarity_matrix_fast(mat, metric, uptr=False, axis=0):
    if metric == 'spearmanr':
        logging.info('Please wait..')
        from scipy.stats import spearmanr
        initial = np.zeros((mat.shape[axis], 
                            mat.shape[axis]))
        for i, vec1 in enumerate(mat):
            for j, vec2 in enumerate(mat):
                initial[i,j] = spearmanr(vec1, vec2).correlation
        mat = initial
    else:
        from scipy.spatial.distance import pdist, squareform
        mat = squareform(pdist(mat, metric))
    if uptr:
        return up_triang(mat, k=1)
    else:
        return mat

def up_triang(mat, k=1):
    return mat[np.triu_indices(len(mat), k)]

def perform_analysis(X, y, setting='pca'):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    X_std = StandardScaler(with_std=False).fit_transform(X)  # standardization of data

    ncomps=3
    if setting == 'pca':
        pca = PCA(n_components=ncomps) 
        X_final = pca.fit_transform(X_std)
        print("Explained Variance with {0:d} components: {1:.2f}".format(ncomps,pca.explained_variance_ratio_.sum()) )
    elif setting == 'lda':
        lda = LDA(n_components=ncomps) 
        X_final = lda.fit_transform(X_std,y)
        print(X_final.shape)

    plt.figure(figsize=(10,8))
    sns.scatterplot(X_final[:,0], X_final[:,1],alpha=0.5, legend='full', hue=y) 
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.show()

def np_hist(df, col, res=0.1, rot=45, fs=12):
    import matplotlib.pyplot as plt
    import numpy as np
    values = df[col]
    _bins, _edges = np.histogram(values, np.arange(df[col].min(), df[col].max(), res))
    plt.plot(_edges[:len(_edges)-1], _bins)
    plt.ylabel('counts'); plt.xlabel(col)
    plt.xticks(rotation=rot, fontsize=fs);
    plt.yticks(fontsize=fs);
    plt.show()

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