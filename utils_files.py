import numpy as np
from wavhandler import get_data, DataSet, BASE_DIR
import librosa
from utils import SR, N_FFT, HOP_LEN, crop_rec
from PIL import Image
import os
from tqdm import tqdm 

def export_to_dir(export_dir=os.path.join(BASE_DIR,'Wingbeats_spectrograms'), 
                    crop=False):
    import warnings
    from sklearn import preprocessing
    X_names, y = get_data(dataset='MOSQUITOES',
                        only_names=True,
                        text_labels=True,
                        nr_signals=np.inf)

    for wav, label in tqdm(zip(X_names, y)):
        data, rate = librosa.load(wav, sr = SR)
        if crop:
            data = crop_rec(data)
        
        data = librosa.stft(data, n_fft=N_FFT, hop_length=HOP_LEN)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = librosa.amplitude_to_db(data) # warnings about using np.abs(data)
        data = np.flipud(data)
        data = scale_arr_0_255(data)

        im = Image.fromarray(data)
        if im.mode != 'RGB':
            im = im.convert('RGB')

        fileName = wav.split('/')[-1][:-4]
        dirName = os.path.join(export_dir, label)
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ")

        filePath = dirName + '/' + fileName + ".jpg"
        if not os.path.exists(filePath):
            im.save(filePath)
    print('Done')

def scale_arr_0_255(arr):
    return (arr - arr.min()) * (1/(arr.max() - arr.min()) * 255).astype('uint8')