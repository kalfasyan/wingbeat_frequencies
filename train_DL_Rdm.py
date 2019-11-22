import sys
sys.path.insert(0,'..')
from wavhandler import *
# from utils_train import train_test_val_split
from pandas.plotting import register_matplotlib_converters
from utils_train import test_inds, test_days
register_matplotlib_converters()
import numpy as np

np.random.seed(42)

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

text_y = y
le = LabelEncoder()
y = le.fit_transform(y.copy())


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

# # Modelling

from utils_train import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization,Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.densenet import DenseNet121
from wavhandler import *
import math
from utils_train import TrainConfiguration

traincf = TrainConfiguration(X=X_train, y=y_train, setting='stft', model_name='Rdm_stft')

targets = 6

current_model = DenseNet121
model = current_model(input_tensor = Input(shape = (129, 120, 1)), 
                      classes = len(traincf.target_names), 
                      weights = None)

# # Build the Neural Network
# model = Sequential()

# model.add(Conv1D(16, 3, activation='relu', input_shape=(5000, 1)))
# model.add(Conv1D(16, 3, activation='relu'))
# model.add(BatchNormalization())

# model.add(Conv1D(32, 3, activation='relu'))
# model.add(Conv1D(32, 3, activation='relu'))
# model.add(BatchNormalization())

# model.add(MaxPooling1D(2))
# model.add(Conv1D(64, 3, activation='relu'))
# model.add(Conv1D(64, 3, activation='relu'))
# model.add(BatchNormalization())

# model.add(MaxPooling1D(2))
# model.add(Conv1D(128, 3, activation='relu'))
# model.add(Conv1D(128, 3, activation='relu'))
# model.add(BatchNormalization())

# model.add(MaxPooling1D(2))
# model.add(Conv1D(256, 3, activation='relu'))
# model.add(Conv1D(256, 3, activation='relu'))
# model.add(BatchNormalization())
# model.add(GlobalAveragePooling1D())

# model.add(Dropout(0.5))
# model.add(Dense(targets, activation='softmax'))

model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

callbacks_list = traincf.callbacks_list

model.fit_generator(train_generator(X_train, y_train, batch_size=traincf.batch_size,
                                   target_names=traincf.target_names,
                                   setting=traincf.setting),
                   steps_per_epoch = int(math.ceil(float(len(X_train)) / float(traincf.batch_size))),
                   epochs=traincf.epochs,
                   validation_data = valid_generator(X_val, y_val,
                                                    batch_size=traincf.batch_size,
                                                    target_names=traincf.target_names,
                                                    setting=traincf.setting),
                    validation_steps=int(math.ceil(float(len(X_test))/float(traincf.batch_size))),
                    callbacks = traincf.callbacks_list)

model.load_weights(traincf.top_weights_path)
y_pred = model.predict_generator(valid_generator(X_test, 
                                                    y_test, 
                                                    batch_size=traincf.batch_size, 
                                                    setting=traincf.setting, 
                                                    target_names=traincf.target_names),
        steps = int(math.ceil(float(len(X_test)) / float(traincf.batch_size))))

# import seaborn as sns
# sns.set()
# import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
x = confusion_matrix(np.array(y_test), np.argmax(y_pred, axis=1))

# plt.figure(figsize=(16,12))
# sns.set(font_scale=1.2)
# ticks = ['Ae. aegypti',' Ae. albopictus','An. arabiensis','An. gambiae','C. pipiens','C. quinquefasciatus']
# ticks_short = ['Ae. aeg','Ae. alb','An. arab','An. gambiae','C. pip','C. quin']
# sns.heatmap(x, annot=True, fmt='.0f', xticklabels=ticks, yticklabels=ticks_short)

print(x)

model.load_weights(traincf.top_weights_path)
loss, acc = model.evaluate_generator(valid_generator(X_test, 
                                                    y_test, 
                                                    batch_size=traincf.batch_size, 
                                                    setting=traincf.setting, 
                                                    target_names=traincf.target_names),
        steps = int(math.ceil(float(len(X_test)) / float(traincf.batch_size))))

print('loss', loss)
print('Test accuracy:', acc)

from sklearn.metrics import balanced_accuracy_score

print('Balanced accuracy:')
print(balanced_accuracy_score(np.array(y_test), np.argmax(y_pred, axis=1)))
