#!/usr/bin/env python
# coding: utf-8

# In[146]:
#get_ipython().run_line_magic('reset', '-f')
from keras.layers import Input, Dense
from keras.models import Model


# In[147]:


# # this is the size of our encoded representations
# encoding_dim = 16  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# # this is our input placeholder
# input_img = Input(shape=(129,))
# # "encoded" is the encoded representation of the input
# encoded = Dense(encoding_dim, activation='relu')(input_img)
# # "decoded" is the lossy reconstruction of the input
# decoded = Dense(129, activation='sigmoid')(encoded)

# # this model maps an input to its reconstruction
# autoencoder = Model(input_img, decoded)
# # this model maps an input to its encoded representation
# encoder = Model(input_img, encoded)
# # create a placeholder for an encoded (32-dimensional) input
# encoded_input = Input(shape=(encoding_dim,))
# # retrieve the last layer of the autoencoder model
# decoder_layer = autoencoder.layers[-1]
# # create the decoder model
# decoder = Model(encoded_input, decoder_layer(encoded_input))
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# In[220]:


from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(129,1))  # adapt this if using `channels_first` image data format

x = Conv1D(64, 3, activation='relu', padding='valid')(input_img)
x = MaxPooling1D((2), padding='valid')(x)
x = Conv1D(32, (3), activation='relu', padding='valid')(x)
x = MaxPooling1D((2), padding='valid')(x)
x = Conv1D(8, (3), activation='relu', padding='valid')(x)
xs = MaxPooling1D((2), padding='valid')(x)
x = Flatten()(xs)
encoded = Dense(8, activation='relu')(x)

x = Conv1D(8, (3), activation='relu', padding='valid')(xs)
x = UpSampling1D((2))(x)
x = Conv1D(32, (3), activation='relu', padding='valid')(x)
x = UpSampling1D((2))(x)
x = Conv1D(64, (3), activation='relu')(x)
x = UpSampling1D((2))(x)
x = Flatten()(x)
decoded = Dense(129, activation='relu')(x)
decoded = Reshape((129,1))(decoded)
#decoded = Conv1D(129, (3), activation='sigmoid', padding='valid')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()


# In[221]:


import numpy as np
import pandas as pd
from sklearn.utils import shuffle

df = pd.read_csv('../data/mosquitos_test.csv', index_col=0)
X = shuffle(df.values, random_state=3).astype('float32')
x_train, x_test = X[:50000,:], X[50000:,:]
print(x_train.shape)
print(x_test.shape)

x_train = np.reshape(x_train, (len(x_train), 129, 1))
x_test = np.reshape(x_test, (len(x_test), 129, 1))

print(x_train.shape)
print(x_test.shape)


# In[ ]:



autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=16,
                shuffle=True,
                validation_data=(x_test, x_test))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


# In[ ]:


# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.plot(x_test[i])#.reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.plot(decoded_imgs[i])#.reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


# 0.0469 loss with bs=128 and encoding_dim=15


# In[ ]:





# In[ ]:





# In[ ]:


























# In[ ]:
