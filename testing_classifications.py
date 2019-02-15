#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reset', '-f')
import glob, os, sys, io
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import pandas as pd
import numpy as np

from wavhandler import *
from utils import *
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
sn.set()

import logging
logger = logging.getLogger()
logger.propagate = False
logger.setLevel(logging.ERROR)
np.random.seed(0)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


#bi_classes = ['LG_drosophila_10_09', 'LG_zapr_26_09']
#target_names = all_6
DATADIR = '/home/kalfasyan/data/insects/Wingbeats/'
target_names = os.listdir(DATADIR)

X_names, y = get_data(filedir= DATADIR,
                      target_names=target_names, nr_signals=np.inf, only_names=True)


# In[4]:


for i,t in enumerate(target_names):
    print(i,t)
print(target_names)


# # Creating a dataframe of PSDs for all mosquito classes

# In[5]:


get_ipython().run_cell_magic('time', '', "X = make_df_parallel(names=X_names, setting='psd_dB')\ndf_concat = pd.DataFrame(X.T)\ndf_concat['label'] = y\n\n# print(df_concat.label.value_counts())\n# df_concat = df_concat[df_concat.label.isin([0,1])]\n# print(df_concat.label.value_counts())")

# ## Selecting which dataframe to use

df = df_concat#df_mosquitos.iloc[:,:-1]
cols = df.columns.tolist()
labels = df.label
classes = np.unique(labels)
#df.label.value_counts()

pd.Series(labels).value_counts()


# # Training a classifier

# X, y = get_data(target_names=target_names, nr_signals=20000, only_names=False)
# X = transform_data(X)

from sklearn.neural_network import MLPClassifier

X, y = shuffle(df.iloc[:,:-1].values, labels, random_state=3)

from imblearn.under_sampling import RandomUnderSampler
ros = RandomUnderSampler(random_state=0)
ros.fit(X,y)
X, y = ros.fit_resample(X,y)
print('After undersampling: \n{}\n'.format(pd.DataFrame(y).iloc[:,0].value_counts()))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

classifier =  MLPClassifier(hidden_layer_sizes=(5,), 
                            alpha=1, 
                            learning_rate='adaptive',
                            verbose=True,
                            random_state=42)
#xgboost.XGBClassifier(n_estimators=650, learning_rate=0.2, n_jobs=-1)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

ac = accuracy_score(y_test, y_pred)
#cv_ac = cross_val_score(classifier, X, y, cv=3, scoring='accuracy')
print("Name: %s, ac: %f" % ('model', ac))
#print("Name: %s, cv_ac: %f" % ('XGBoost', np.mean(cv_ac)))


cm = confusion_matrix(y_test, y_pred)
print(cm)

print('done')