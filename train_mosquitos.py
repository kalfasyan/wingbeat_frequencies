import pandas as pd
import numpy as np
from wavhandler import *
from utils_train import *

data = Dataset('Wingbeats')
data.read(data='all', setting='read', labels='text', loadmat=False)

# data1 = Dataset('Leafminers')
# data1.read(data=data1.target_classes[0], setting='read', loadmat=False, labels='text')

# data2 = Dataset('LG')
# data2.read(data=data2.target_classes[0], setting='read', loadmat=False, labels='text')

# data3 = Dataset('LG')
# data3.read(data=data3.target_classes[1], setting='read', loadmat=False, labels='text')

# data4 = Dataset('Pcfruit')
# data4.read(data=data4.target_classes[1], setting='read', loadmat=False, labels='text')

# # data1.clean(plot=False)
# # data2.clean(plot=False)
# # data3.clean(plot=False)
# # data4.clean(plot=False)

# data = pd.DataFrame()
# data['filenames'] = pd.concat([data1.filenames, data2.filenames, data3.filenames, data4.filenames], axis=0).reset_index(drop=True)
# data['y'] = data.filenames.apply(lambda x: x.split('/')[6])

make_classification_conv2d(data.filenames.tolist(), data.y.tolist(), model_name='mosquitoes_buda', setting='stft',
                          undersampling=False)
# make_classification_conv2d(data1.filenames.tolist(), data1.y.tolist(), setting='melspec', undersampling=False)