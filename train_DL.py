from wavhandler import Dataset
import numpy as np
import sys
import math
from utils import TEMP_DATADIR
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from utils_train import train_test_val_split, TrainConfiguration, train_generator, valid_generator,mosquito_data_split, train_model_dl
import deepdish as dd
seed = 42
np.random.seed(seed=seed)



splitting = sys.argv[1]
data_setting = sys.argv[2]
model_setting = sys.argv[3]

assert splitting in ['random','randomcv','custom'], "Wrong splitting method given."
assert data_setting in ['raw','stft','psd_dB', 'cwt', 'rawflt','stftflt','psd','psdflt','psd_dBflt'], "Wrong data settting given."
assert model_setting in ['wavenet','lstm','gru','conv1d','conv1d_psd',
                        'DenseNet121','DenseNet169','DenseNet201',
                        'InceptionResNetV2','VGG16','VGG19',
                        'dl4tsc_fcn','dl4tsc_res', 'tsc_res_baseline',
                        'tsc_fcn_baseline', 'conv1d_baseline', 'dl4tsc_inc'], "Wrong model setting given"

data = Dataset('Wingbeats')
print(data.target_classes)

print(f'SPLITTING DATA {splitting}')
X_train, X_val, X_test, y_train, y_val, y_test = mosquito_data_split(splitting=splitting, dataset=data, downsampling=True)

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
                            flag=f'{i}')
        results[f'{splitting}_{i}'] = res

dd.io.save(f'{TEMP_DATADIR}/{splitting}_{data_setting}_{model_setting}_results.h5', 
            {f'results': results})