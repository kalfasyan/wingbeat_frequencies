from wavhandler import Dataset
import numpy as np
import sys
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from utils_train import train_test_val_split, TrainConfiguration, train_generator, valid_generator,mosquito_data_split, train_model_dl
seed = 42
np.random.seed(seed=seed)



splitting = sys.argv[1]
data_setting = sys.argv[2]
model_setting = sys.argv[3]
cnn_if_2d = sys.argv[4]

assert splitting in ['random','randomcv','custom'], "Wrong splitting method given."
assert data_setting in ['raw','stft','psd_dB'], "Wrong data settting given."
assert model_setting in ['wavenet','lstm','gru','LSTM','GRU','CONV1D','CONV2D','conv1d','conv2d','conv1d_psd','CONV1D_psd']

data = Dataset('Wingbeats')
print(data.target_classes)

print(f'SPLITTING DATA {splitting}')
X_train, X_val, X_test, y_train, y_val, y_test = mosquito_data_split(splitting=splitting, dataset=data)

results = {}
if splitting in ['random', 'randomcv']:
    res = train_model_dl(dataset=data,
                        model_setting=model_setting,
                        splitting=splitting, 
                        data_setting=data_setting,
                        X_train=X_train, 
                        y_train=y_train, 
                        X_val=X_val, 
                        y_val=y_val, 
                        X_test=X_test, 
                        y_test=y_test,
                        cnn_if_2d=cnn_if_2d,
                        flag='na')
    results[splitting] = res
elif splitting == 'custom':
    for i in range(5):
        xtrain, ytrain = shuffle(X_train[i], y_train[i], random_state=seed)
        xval, yval = shuffle(X_val[i], y_val[i], random_state=seed)
        res = train_model_dl(dataset=data,
                            model_setting=model_setting,
                            splitting=splitting, 
                            data_setting=data_setting,
                            X_train=xtrain, 
                            y_train=ytrain, 
                            X_val=xval, 
                            y_val=yval, 
                            X_test=X_test, 
                            y_test=y_test,
                            cnn_if_2d=cnn_if_2d,
                            flag=f'split_{i}')
        results[f'{splitting}_{i}'] = res

import deepdish as dd
dd.io.save(f'temp_data/{splitting}_{data_setting}_{model_setting}_results.h5', 
            {f'results': results})