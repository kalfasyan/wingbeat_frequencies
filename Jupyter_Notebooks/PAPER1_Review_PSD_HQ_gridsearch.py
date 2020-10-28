#!/usr/bin/env python
# coding: utf-8
import sys

sys.path.insert(0,'..')
import math
import multiprocessing

import deepdish as dd
import matplotlib.pyplot as plt
import numpy as np
from utils import TEMP_DATADIR
import pandas as pd
from joblib import dump, load
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, log_loss, make_scorer)
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utils_train import (TrainConfiguration, mosquito_data_split,
                         train_generator, train_model_ml, train_test_val_split,
                         valid_generator)
from wavhandler import *
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-s", "--splitting", required=True, help="name of the user")
ap.add_argument("-d", "--datasetting", required=True, help="name of the user")
ap.add_argument("-m", "--modelsetting", required=True, help="name of the user")
ap.add_argument("-t", "--test", required=False, help="name of the user")
args = vars(ap.parse_args())

n_cpus = multiprocessing.cpu_count()

seed = 42
np.random.seed(seed=seed)


splitting = "random" # We perform grid search with random splitting always
data_setting = args['datasetting'] #"psdHQ"
model_setting = args['modelsetting'] #"knn"
test = args['test']

assert splitting in ['random','randomcv','custom'], "Wrong splitting method given."
assert data_setting in ['raw','psd_dB','psdHQ'], "Wrong data settting given."
assert model_setting in ['knn','randomforest','xgboost']

data = Dataset('Wingbeats')
print(data.target_classes)

print(f'SPLITTING DATA {splitting}')
X_train, X_val, X_test, y_train, y_val, y_test = mosquito_data_split(splitting=splitting, dataset=data)
print("done")

x_test = make_df_parallel(names=X_test, setting=data_setting).values
print("xtest created")

# ## if "RANDOM"
results = {}
X_train.extend(X_val)
y_train.extend(y_val)

if test == 'True':
    X_train = X_train[:5000]
    y_train = y_train[:5000]

x_train = make_df_parallel(names=X_train, setting=data_setting).values
print('xtrain created')
x_val = make_df_parallel(names=X_val, setting=data_setting).values
print('xval created')

if model_setting == 'knn':
    estimator = KNeighborsClassifier(n_neighbors=11, weights='uniform',metric='manhattan', n_jobs=-1)
    parameters = {'n_neighbors':(7,9,11,13,15,17), 'weights':('uniform','distance'), 'metric': ('manhattan','euclidean')}
elif model_setting == 'randomforest':
    estimator = RandomForestClassifier()
    parameters = {'max_depth': (None, 5),
                'min_samples_leaf': (3,4),
                'min_samples_split': (8,10),
                'max_features': ('auto','sqrt','log2'),
                'criterion': ('gini', 'entropoy'),
                'n_estimators': (450, 550, 1000)}
elif model_setting == 'xgboost':
    estimator = XGBClassifier()
    parameters = {'max_depth': (4,5),
                    'learning_rate': (0.2, 0.3, 0.4),
                    'gamma': (0, 0.5),
                    'n_estimators': (450,550,1000)}

clf = GridSearchCV(estimator, parameters, n_jobs=-1, verbose=1)
print('running grid search')
clf.fit(x_train, y_train)
print('done')

estimator = clf.best_estimator_
print('running cv with best model')

cvfolds = 5
cv_results = cross_validate(estimator, x_train, y_train, cv=cvfolds, 
                            return_estimator=True, 
                            return_train_score=True, 
                            scoring=make_scorer(balanced_accuracy_score),
                            verbose=1, 
                            n_jobs=-1)

# CREATING RESULTS
y_preds = [cv_results['estimator'][i].predict(x_test) for i in range(cvfolds)]
y_pred_probas = [cv_results['estimator'][i].predict_proba(x_test) for i in range(cvfolds)]

cms = [confusion_matrix(y_test, y_preds[i]) for i in range(cvfolds)]
b_accs = [balanced_accuracy_score(y_test, y_preds[i]) for i in range(cvfolds)]
logloss = [log_loss(y_test, y_pred_probas[i]) for i in range(cvfolds)]
clf_reports = [classification_report(y_test, y_preds[i], target_names=data.target_classes) for i in range(cvfolds)]

results['y_pred_probas'] = y_pred_probas
results['cms'] = cms
results['b_accs'] = b_accs
results['logloss'] = logloss
results['clf_reports'] = clf_reports
results['train_score'] = cv_results['train_score']
results['val_score'] = cv_results['val_score']
results['test_score'] = cv_results['test_score']
results['model'] = cv_results['estimator']
results['gs_bestmodel'] = clf.best_estimator_
results['gs_bestscore'] = clf.best_score_
results['gs_bestparams'] = clf.best_params_

dd.io.save(f'{TEMP_DATADIR}/{splitting}_{data_setting}_{model_setting}_results.h5', 
            {f'results': results})
