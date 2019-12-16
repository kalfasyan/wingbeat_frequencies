from wavhandler import Dataset, make_df_parallel
import numpy as np
import sys
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from utils_train import train_test_val_split, TrainConfiguration, train_generator
from utils_train import valid_generator,mosquito_data_split, train_model_ml

seed = 42
np.random.seed(seed=seed)

splitting = sys.argv[1]
data_setting = sys.argv[2]
model_setting = sys.argv[3]

assert splitting in ['random','randomcv','custom'], "Wrong splitting method given."
assert data_setting in ['raw','psd_dB'], "Wrong data settting given."
assert model_setting in ['knn','randomforest','xgboost']

data = Dataset('Wingbeats')
print(data.target_classes)

print(f'SPLITTING DATA {splitting}')
X_train, X_val, X_test, y_train, y_val, y_test = mosquito_data_split(splitting=splitting, dataset=data)
x_test = make_df_parallel(names=X_test, setting=data_setting).values

if splitting in ['random', 'randomcv']:
    X_train.extend(X_val)
    y_train.extend(y_val)
    x_train = make_df_parallel(names=X_train, setting=data_setting).values
    x_val = make_df_parallel(names=X_val, setting=data_setting).values

    model = train_model_ml(dataset=data,
                            model_setting=model_setting,
                            splitting=splitting, 
                            data_setting=data_setting,
                            x_train=x_train, 
                            y_train=y_train, 
                            x_val=x_val, 
                            y_val=y_val, 
                            x_test=x_test, 
                            y_test=y_test,
                            flag='ML')
elif splitting == 'custom':
    from joblib import dump, load
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.model_selection import cross_val_score

    for i in range(5):
        x_train_fold = make_df_parallel(names=X_train[i], setting=data_setting).values
        x_val_fold = make_df_parallel(names=X_val[i], setting=data_setting).values
        estimator = train_model_ml(dataset=data,
                                model_setting=model_setting,
                                splitting=splitting, 
                                data_setting=data_setting,
                                x_train=x_train_fold, 
                                y_train=y_train[i], 
                                x_val=x_val_fold, 
                                y_val=y_val[i], 
                                x_test=x_test, 
                                y_test=y_test,
                                flag=f'split_{i}')

        y_train[i].extend(y_val[i])
        estimator.fit(np.vstack((x_train_fold, x_val_fold)), y_train[i])

        y_pred = np.argmax( estimator.predict_proba(x_test) , axis=1)
        cm = confusion_matrix(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred, target_names=data.target_classes)

        dump(estimator, f'temp_data/{splitting}_{data_setting}_{model_setting}_{bacc:.2f}_split_{i}_test.joblib') 
        with open(f'temp_data/{splitting}_{data_setting}_{model_setting}_split_{i}_test_results.txt', "a+") as resultsfile:
            resultsfile.write(f'classifier: {estimator}, \nbal_acc:{bacc}, \n{cm}\n{clf_report}\n')
