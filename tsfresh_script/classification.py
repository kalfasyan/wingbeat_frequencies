import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from utils import *

sampling = sys.argv[2]
# SETTINGS
model = sys.argv[1]
print('Running classification with {}\n'.format(model))
dataset = pd.read_pickle('./data/mosquitos.pkl')

# DEFINITION OF X, y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print('Class balance: \n{}\n'.format(pd.DataFrame(y).iloc[:,0].value_counts()))

if sampling == 'under':	# UNDERSAMPLING to match lowest class samples
	from imblearn.under_sampling import RandomUnderSampler
	ros = RandomUnderSampler(random_state=0)
	ros.fit(X,y)
	X, y = ros.fit_resample(X,y)
	print('After undersampling: \n{}\n'.format(pd.DataFrame(y).iloc[:,0].value_counts()))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

if sampling == 'over': # OVERSAMPLING only on training data
	from imblearn.over_sampling import SMOTE
	sm = SMOTE(random_state=0, sampling_strategy='auto', n_jobs=-1)
	X_train, y_train = sm.fit_sample(X_train, y_train)
	print('After over-sampling: \n{}\n'.format(pd.DataFrame(y_train).iloc[:,0].value_counts()))

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Selecting a Classifier and fit it to Training data
classifier = create_classifier(model)
classifier.fit(X_train, y_train)

# Saving a dataframe with feature importances if applicable to classifier
try:
	feature_importances = pd.DataFrame(classifier.feature_importances_,
										index = dataset.columns.tolist()[:-1],
										columns=['importance']).sort_values('importance', ascending=False)
	print(feature_importances)
except:
	print('No feature importances for this classifier!')

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy based on 1 split of train/test: %.3f" %accuracy_score(y_test, y_pred))

# Calculating cross-validation score
print("Calculating cross-validation score")
from sklearn.model_selection import cross_val_score
crossval_score = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
print(np.round(np.mean(crossval_score),3))

# Making a dataframe of the confusion matrix to plot it
df_cm = pd.DataFrame(cm, index=[i for i in dataset.label1.unique()], 
					columns=[i for i in dataset.label1.unique()])
plt.figure(figsize=(12,7))
sn.heatmap(df_cm, annot=True, fmt='g').\
			set_title('Mean CV Accuracy Score: '+\
			str(np.round(np.mean(crossval_score),3)))

plt.show()

