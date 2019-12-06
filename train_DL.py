from wavhandler import *
import numpy as np
import sys
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from utils_train import train_test_val_split, TrainConfiguration, train_generator, valid_generator,mosquito_data_split, train_model

np.random.seed(42)



splitting = sys.argv[1]
data_setting = sys.argv[2]
model_setting = sys.argv[3]
cnn_if_2d = sys.argv[4]

assert splitting in ['random','randomcv','custom'], "Wrong splitting method given."
assert data_setting in ['raw','stft'], "Wrong data settting given."
assert model_setting in ['wavenet','lstm','gru','LSTM','GRU','CONV1D','CONV2D','conv1d','conv2d']

data = Dataset('Wingbeats')
print(data.target_classes)

print(f'SPLITTING DATA {splitting}')
X_train, X_val, X_test, y_train, y_val, y_test = mosquito_data_split(splitting=splitting, data=data)

model = train_model(model_setting=model_setting,
                    splitting=splitting, 
                    data_setting=data_setting,
                    X_train=X_train, 
                    y_train=y_train, 
                    X_val=X_val, 
                    y_val=y_val, 
                    X_test=X_test, 
                    y_test=y_test,
                    cnn_if_2d=cnn_if_2d,
                    flag='data_centering')