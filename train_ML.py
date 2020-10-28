from wavhandler import Dataset, make_df_parallel
import numpy as np
import sys
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from utils_train import train_test_val_split, TrainConfiguration, train_generator
from utils_train import valid_generator,mosquito_data_split, train_model_ml
import deepdish as dd
from joblib import dump, load
from utils import TEMP_DATADIR
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, log_loss
from sklearn.model_selection import cross_val_score
seed = 42
np.random.seed(seed=seed)

splitting = sys.argv[1]
data_setting = sys.argv[2]
model_setting = sys.argv[3]

assert splitting in ['random','randomcv','custom'], "Wrong splitting method given."
assert data_setting in ['raw','psd_dB','psdHQ'], "Wrong data settting given."
assert model_setting in ['knn','randomforest','xgboost']

print(f"Splitting: {splitting}, Data setting: {data_setting}, Model setting: {model_setting}")

data = Dataset('Wingbeats')
print(data.target_classes)

print(f'SPLITTING DATA {splitting}')
X_train, X_val, X_test, y_train, y_val, y_test = mosquito_data_split(splitting=splitting, dataset=data)
print("Creating test dataframe")
x_test = make_df_parallel(names=X_test, setting=data_setting).values
print("Test dataframe created.")
results = {}
if splitting in ['random', 'randomcv']:
    X_train.extend(X_val)
    y_train.extend(y_val)
    print("Creating train dataframe")
    x_train = make_df_parallel(names=X_train, setting=data_setting).values
    print("Train dataframe created.")
    print("Creating val dataframe")
    x_val = make_df_parallel(names=X_val, setting=data_setting).values
    print("Val dataframe created.")

    print("Training model..")
    _, res = train_model_ml(dataset=data,
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
    results[splitting] = res
    print("Finished training.")

elif splitting == 'custom':
    train_scores, val_scores, cms, test_scores, logloss, clf_reports, y_preds, y_pred_probas = [],[],[],[],[],[],[],[]
    for i in range(5):
        print(f"Creating train dataframe for fold {splitting}_{i}")
        x_train_fold = make_df_parallel(names=X_train[i], setting=data_setting).values
        print(f"Train dataframe created.")
        print(f"Creating val dataframe for fold {splitting}_{i}")
        x_val_fold = make_df_parallel(names=X_val[i], setting=data_setting).values
        print(f"Val dataframe created.")

        print("Training model..")
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
        print(f"Finished Training fold {i}.\nNow saving results..")
        y_preds.append( estimator.predict(x_test) )
        y_pred_probas.append( estimator.predict_proba(x_test) )
        train_scores.append( balanced_accuracy_score(y_train[i], estimator.predict(x_train_fold)) )
        val_scores.append( balanced_accuracy_score(y_val[i], estimator.predict(x_val_fold)) )
        cms.append(confusion_matrix(y_test, y_preds[i]))
        test_scores.append(balanced_accuracy_score(y_test, y_preds[i]))
        logloss.append(log_loss(y_test, y_pred_probas[i]))
        clf_reports.append(classification_report(y_test, y_preds[i], target_names=data.target_classes))
        print(f"Gathered all results for fold {splitting}_{i}")
    results['y_preds'] = y_preds
    results['y_pred_probas'] = y_pred_probas
    results['y_test'] = y_test
    results['cms'] = cms
    results['logloss'] = logloss
    results['clf_reports'] = clf_reports
    results['train_score'] = train_scores
    results['val_score'] = val_scores
    results['test_score'] = test_scores
    results['logloss_test'] = logloss
    results['model'] = estimator
    
print("Procedure completed. Saving all results.")
dd.io.save(f'{TEMP_DATADIR}/{splitting}_{data_setting}_{model_setting}_results.h5', 
            {f'results': results})
print("Finished.")