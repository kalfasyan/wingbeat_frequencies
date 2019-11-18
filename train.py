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


model = sys.argv[1]
print(f"RUNNING CLASSIFICATION WITH {model} MODEL")

data = Dataset('Wingbeats')
print(data.target_classes)

data.read('Ae. aegypti', loadmat=False)
x1 = data.filenames.sample(14800)
data.read('Ae. albopictus', loadmat=False)
x2 = data.filenames.sample(14800)
data.read('An. arabiensis', loadmat=False)
x3 = data.filenames.sample(14800)
data.read('An. gambiae', loadmat=False)
x4 = data.filenames.sample(14800)
data.read('C. pipiens', loadmat=False)
x5 = data.filenames.sample(14800)
data.read('C. quinquefasciatus', loadmat=False)
x6 = data.filenames.sample(14800)

X = pd.concat([x1, x2, x3, x4, x5, x6], axis=0)
y = X.apply(lambda x: x.split('/')[len(BASE_DIR.split('/'))])

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.utils import shuffle
from utils_train import train_test_val_split

X,y = shuffle(X.tolist(),y.tolist(), random_state=0)

X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(X,y,test_size=0.13514, val_size=0.2)


keys = pd.Series(le.inverse_transform(y_train)).value_counts().index.tolist()
values = pd.Series(y_train).value_counts().index.tolist()
mapping = dict(zip(keys, values))
print(sorted(mapping.items(), key=lambda x: x[1]))
vcounts = pd.Series(y_train).value_counts()
vcounts.index = mapping.keys()
print(vcounts)

t = time.time()
print("Reading data into dataframes in parallel.")
df_train = make_df_parallel(setting='psd_dB', names=X_train+X_val)
# df_val = make_df_parallel(setting='psd_dB', names=X_val)
df_test = make_df_parallel(setting='psd_dB', names=X_test)
print(f"{time.time() - t} seconds")

y_train = y_train + y_val

if model.startswith('knn'):
    # SCALING DATA
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()#with_std=True)
    x_train = sc.fit_transform(df_train.values)
    x_test = sc.fit_transform(df_test.values)


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
dump(classifier, f'{model}_{ac:.2f}.joblib') 
print(f"Saved model in {time.time() - t} seconds")