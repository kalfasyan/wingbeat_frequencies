#!/usr/bin/env python
# coding: utf-8

import sys
from wavhandler import *
from pandas.plotting import register_matplotlib_converters
from utils_train import test_inds, test_days
register_matplotlib_converters()
import numpy as np
import time

np.random.seed(42)

model = sys.argv[1] +'_impVal_'
setting = sys.argv[2]
assert setting in ['psd_dB','raw'], 'Wrong setting given'

print(f"RUNNING CLASSIFICATION WITH {model} MODEL and {setting} setting")

data = Dataset('Wingbeats')
print(data.target_classes)

def train_test_filenames(dataset, species, train_dates=[], test_dates=[]):
    dataset.read(species, loadmat=False)
    dataset.get_sensor_features()
    sub = dataset.df_features
    # sub.groupby('datestr')['filenames'].count().plot(kind="bar")
    print(sub['datestr'].unique().tolist())

    test_fnames = sub[sub.datestr.isin(test_dates)].filenames
    if len(train_dates): # if train dates are given
        train_fnames = sub[sub.datestr.isin(train_dates)].filenames
    else:
        train_fnames = sub[~sub.datestr.isin(test_dates)].filenames

    print("{} train filenames, {} test filenames".format(train_fnames.shape[0], test_fnames.shape[0]))
    return train_fnames, test_fnames

# ### Ae. Aegypti
x1_tr, x1_ts = train_test_filenames(data,'Ae. aegypti', test_dates=['20161213','20161212'])
# ### Ae. albopictus
x2_tr, x2_ts = train_test_filenames(data,'Ae. albopictus', test_dates=['20170103', '20170102'])
# ### An. arabiensis
x3_tr, x3_ts = train_test_filenames(data,'An. arabiensis', test_dates=['20170319','20170320',
                                                                       '20170318','20170317'], 
                                                            train_dates=['20170201','20170202', '20170203','20170204',
                                                                        '20170205','20170206','20170131','20170130'])
# ### An. gambiae
x4_tr, x4_ts = train_test_filenames(data,'An. gambiae', test_dates=['20170110', '20170109']) 
# ### Culex quinquefasciatus
x5_tr, x5_ts = train_test_filenames(data,'C. quinquefasciatus', test_dates=['20161219']) 
# ### Culex pipiens
x6_tr, x6_ts = train_test_filenames(data,'C. pipiens', test_dates=['20161206', '20161205']) 

x1_tr, x1_ts = x1_tr.sample(12800), x1_ts.sample(2000)
x2_tr, x2_ts = x2_tr.sample(12800), x2_ts.sample(2000)
x3_tr, x3_ts = x3_tr.sample(12800), x3_ts.sample(2000)
x4_tr, x4_ts = x4_tr.sample(12800), x4_ts.sample(2000)
x5_tr, x5_ts = x5_tr.sample(12800), x5_ts.sample(2000)
x6_tr, x6_ts = x6_tr.sample(12800), x6_ts.sample(2000)

# ## Creating TRAIN/VAL/TEST sets
X_train = pd.concat([x1_tr, x2_tr, x3_tr, x4_tr, x5_tr, x6_tr], axis=0)
X_test = pd.concat([x1_ts, x2_ts, x3_ts, x4_ts, x5_ts, x6_ts], axis=0)
y_train = X_train.apply(lambda x: x.split('/')[len(BASE_DIR.split('/'))])
y_test = X_test.apply(lambda x: x.split('/')[len(BASE_DIR.split('/'))])

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
X_test = X_test.tolist()

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
X_train,y_train = shuffle(X_train.tolist(),y_train.tolist(), random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)


keys = pd.Series(le.inverse_transform(y_train)).value_counts().index.tolist()
values = pd.Series(y_train).value_counts().index.tolist()
mapping = dict(zip(keys, values))
print(sorted(mapping.items(), key=lambda x: x[1]))
vcounts = pd.Series(y_train).value_counts()
vcounts.index = mapping.keys()
print(vcounts)

t = time.time()
print("Reading data into dataframes in parallel.")
df_train = make_df_parallel(setting=setting, names=X_train+X_val)
df_test = make_df_parallel(setting=setting, names=X_test)
print(f"{time.time() - t} seconds")

y_train = y_train + y_val

if model.startswith('knn'):
    # SCALING DATA
    # from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler(with_std=False)
    x_train = df_train.values #sc.fit_transform(df_train.values)
    x_test = df_test.values #sc.fit_transform(df_test.values)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV, cross_val_score

    grid_params = {
        'n_neighbors': [11, 13, 15, 17],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    print(f'Using knn parameters:\n {grid_params}')

    gs = GridSearchCV(KNeighborsClassifier(),
                    grid_params,
                    verbose=1,
                    cv=5,
                    n_jobs=-1)

    t = time.time()
    gs_results = gs.fit(x_train, y_train)

    classifier = gs_results.best_estimator_
elif model.startswith('xgboost'):
    x_train = df_train.values
    x_test = df_test.values

    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier

    parameters = {'max_depth': range (3, 5, 1),
                'n_estimators': range(340, 460, 40),
                'learning_rate': [0.3, 0.4, 0.5],
                'gamma': [0, 0.5]}

    print(f'Using xgboost parameters:\n {parameters}')

    estimator = XGBClassifier(param_grid=parameters,
                            random_state=0,
                            seed=42,
                            verbose=True)

    gs_xgb = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        n_jobs = -1,
        cv = 5,
        verbose=True)

    t = time.time()
    gs_xgb.fit(x_train, y_train)

    classifier = gs_xgb.best_estimator_
else:
    raise NotImplementedError('Not implemented yet.')

print(classifier)
print(f"Ran grid search in {time.time() - t} seconds")

t = time.time()
print("Fitting model in training data")
classifier.fit(x_train, y_train)
y_pred = classifier.predict_proba(df_test.values)
print(f"Fit model in {time.time() - t} seconds")

y_pred = np.argmax(y_pred, axis=1)

from sklearn.metrics import confusion_matrix,balanced_accuracy_score

# Calculating confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

ac = balanced_accuracy_score(y_test, y_pred)
print(f'Balanced Accuracy Score: {ac}')

t = time.time()
print("Saving model")
from joblib import dump, load
dump(classifier, f'temp_data/{model}_{ac:.2f}.joblib') 
print(f"Saved model in {time.time() - t} seconds")