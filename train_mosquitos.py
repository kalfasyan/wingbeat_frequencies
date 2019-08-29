import pandas as pd
import numpy as np
from wavhandler import *
from utils_train import *

data1 = Dataset('Wingbeats')
data1.read(data='all', setting='read', labels='text', loadmat=False)

make_classification_conv2d(data1.filenames.tolist(), data1.y.tolist(), model_name='mosquitos', setting='stft',
                          undersampling=False)