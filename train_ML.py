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
    from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, log_loss
    from sklearn.model_selection import cross_val_score

    train_scores, val_scores, cms, b_accs, logloss, clf_reports = [],[],[],[],[],[]
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

        y_preds = estimator.predict(x_test)
        y_pred_probas = estimator.predict_proba(x_test)

        train_scores.append( balanced_accuracy_score(y_train[i], estimator.predict(x_train_fold)) )
        val_scores.append( balanced_accuracy_score(y_val[i], estimator.predict(x_val_fold)) )
        cms.append(confusion_matrix(y_test, y_preds))
        b_accs.append(balanced_accuracy_score(y_test, y_preds))
        logloss.append(log_loss(y_test, y_pred_probas))
        clf_reports.append(classification_report(y_test, y_preds, target_names=data.target_classes))

        with open(f'temp_data/{splitting}_{data_setting}_{model_setting}_results.txt', "a+") as resultsfile:
            resultsfile.write(f'\n\n\t\tFOLD #: {i}\n '
                                f'train_score: {train_scores[i]},' 
                                f'val_score: {val_scores[i]},' 
                                f'balanced_accuracy_on_test: {b_accs[i]}\n,' 
                                f'log_loss_on_test: {logloss[i]}\n,' 
                                f'confusion_matrix:\n{cms[i]}\n' 
                                f'classification_report:\n{clf_reports[i]}\n')

    mean_train_score = np.mean(train_scores)
    mean_val_score = np.mean(val_scores)
    mean_test_score = np.mean(b_accs)

    with open(f'temp_data/{splitting}_{data_setting}_{model_setting}_results.txt', "a+") as resultsfile:
        resultsfile.write(f'mean_train_score: {mean_train_score},'
                            f'mean_val_score: {mean_val_score},'
                            f'mean_test_score: {mean_test_score}\n') 
