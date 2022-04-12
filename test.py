# %%
from operator import itemgetter
from itertools import groupby
import os
import json
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import audiomentations as am

import audiomentations as am
import noisereduce as nr
import librosa
import scipy.io.wavfile as wav

from dataset import annotate_dataset
from preprocess import augment_signal, represent_dataset, represent_signal, segment_dataset, segment_signal, pipeline_signal
from loaders import load_mat_representation

# %%
SAMPLE_RATE = 24000
SEGMENT_LENGTH = 1
OVERLAP_SIZE = 0

MFCC_COEFF = 26
MFCC_N_FFT = 2048
MFCC_HOP_LENGTH = 512

TRAIN_TRANSFORM = [
    am.AddGaussianSNR(min_snr_in_db=24, max_snr_in_db=40, p=0.8),
    am.HighPassFilter(min_cutoff_freq=60, max_cutoff_freq=100, p=0.8),
    am.LowPassFilter(min_cutoff_freq=3400, max_cutoff_freq=4000, p=0.8),
    am.TimeStretch(min_rate=0.75, max_rate=2,
                   leave_length_unchanged=False, p=0.5),
]

BASE_TRANSFORM = [
    am.Trim(top_db=20, p=1),
    am.Normalize(p=1),
]
# %%
annotate_dataset(f'/src/datasets/base_portuguese',
                 f'/src/tcc/dataset/base_portuguese_4',
                 SAMPLE_RATE,
                 [45, 46, 76, 85])
# %%

segment_dataset(f'/src/tcc/dataset/base_portuguese_4',
                f'/src/tcc/dataset/base_portuguese_4/SEG_1_OVERLAP_0_AUG_30',
                base_trans=BASE_TRANSFORM,
                overlap_size=OVERLAP_SIZE,
                segment_length=SEGMENT_LENGTH)

# %%
mat_dict_test = represent_dataset(f'/src/tcc/dataset/base_portuguese_4/SEG_1_OVERLAP_0_AUG_30',
                                  f'/src/tcc/dataset/base_portuguese_4/SEG_1_OVERLAP_0_AUG_30/MFCC_{MFCC_COEFF}',
                                  n_mfcc=MFCC_COEFF,
                                  n_fft=MFCC_N_FFT,
                                  hop_length=MFCC_HOP_LENGTH)
# %%
mtest = np.mean(np.mean(mat_dict_test['representation'], axis=1), axis=0)
plt.plot(mtest)
plt.show()
# %%
audioment = am.Compose([*BASE_TRANSFORM, *TRAIN_TRANSFORM])
# %%
# mat_dict_valid = load_mat_representation(
#     f'/src/tcc/dataset/base_portuguese_20/SEG_1_OVERLAP_0_AUG_30/MFCC_26/valid/representation.mat')

# mvalid = np.mean(np.mean(mat_dict_valid['representation'], axis=1), axis=0)
# plt.plot(mvalid)
# plt.show()
# %%

mat_dict_test = load_mat_representation(
    f'/src/tcc/dataset/base_portuguese_20/SEG_1_OVERLAP_0_AUG_30/MFCC_26/test/representation.mat')

mtest = np.mean(np.mean(mat_dict_test['representation'], axis=1), axis=0)
plt.plot(mtest)
plt.show()

# %%
X_inf = None
y_inf = None

df = pd.read_csv(f'/src/tcc/dataset/inference/base_portuguese_20/metadata.csv')

for index, row in df.iterrows():
    X_inf_aux, _, _ = pipeline_signal(f'/src/tcc/dataset/inference/base_portuguese_20/audio/{row["filename"]}',
                                      sample_rate=24000,
                                      segment_length=1,
                                      overlap_size=0,
                                      transformations=BASE_TRANSFORM,
                                      n_mfcc=26,
                                      n_fft=2048,
                                      hop_length=512)

    y_inf_aux = [row['label']] * len(X_inf_aux)

    if X_inf is None:
        X_inf = X_inf_aux
        y_inf = y_inf_aux
    elif X_inf.shape[1:] == X_inf_aux.shape[1:]:
        X_inf = np.concatenate((X_inf, X_inf_aux))
        y_inf = np.concatenate((y_inf, y_inf_aux))

# %%
minf = np.mean(np.mean(X_inf, axis=1), axis=0)
plt.plot(minf)
plt.show()
# %%

sig1, _ = librosa.load(
    '/src/tcc/dataset/base_portuguese_4/audio/p1d5cc8d09606418688a9b14370be7960-s03-a01.wav', SAMPLE_RATE)

noise_reduced = nr.reduce_noise(sig1, sr=SAMPLE_RATE)

signals = augment_signal(sig1, SAMPLE_RATE, BASE_TRANSFORM)

sio.wavfile.write('/src/tcc/base.wav', SAMPLE_RATE, sig1)
sio.wavfile.write('/src/tcc/base_noise.wav', SAMPLE_RATE, noise_reduced)

# %%
all_files = []
path = '/src/tcc/models/base_portuguese_4'

for root, dirs, files in os.walk(path):
    for file in files:
        relativePath = os.path.relpath(root, path)
        if relativePath == ".":
            relativePath = ""
        all_files.append(
            (relativePath.count(os.path.sep),
             relativePath,
             file
             )
        )

all_files.sort(reverse=True)

dirs = []

for (count, folder), files in groupby(all_files, itemgetter(0, 1)):
    if folder.find("inference") != -1:
        dirs.append(folder)

    # for file in files:
    #     print('File:', file[2])

# print(' '.join(dirs))

# %%
df_dict = {
    'classes': [],
    'segment_length': [],
    'sample_rate': [],
    'overlap_size': [],
    'augment_size': [],
    'n_mfcc': [],
    'inf_micro': [],
    'inf_macro': [],
    'test_loss': [],
    'test_acc': [],
    'valid_loss': [],
    'valid_acc': [],
    'train_loss': [],
    'train_acc': [],
}


def format_float(f):
    return float(f'{f:.4f}')


for f in dirs:
    overview = open(
        f'{path}/{f}/overview.json', 'r').read()
    overview = json.loads(overview)

    df_dict['classes'].append(len(overview['classes']))
    df_dict['sample_rate'].append(overview['sample_rate'])
    df_dict['segment_length'].append(overview['segment_length'])
    df_dict['overlap_size'].append(overview['overlap_size'])
    df_dict['augment_size'].append(overview['augment_size'])
    df_dict['n_mfcc'].append(overview['representation']['n_mfcc'])
    df_dict['inf_micro'].append(format_float(overview['f1_micro']))
    df_dict['inf_macro'].append(format_float(overview['f1_macro']))
    df_dict['test_loss'].append(
        format_float(overview['scores']['test_loss']))
    df_dict['test_acc'].append(
        format_float(overview['scores']['test_acc']))
    df_dict['valid_loss'].append(
        format_float(overview['scores']['valid_loss']))
    df_dict['valid_acc'].append(
        format_float(overview['scores']['valid_acc']))
    df_dict['train_loss'].append(
        format_float(overview['scores']['train_loss']))
    df_dict['train_acc'].append(
        format_float(overview['scores']['train_acc']))


# %%
pd.DataFrame.from_dict(df_dict).to_csv(f'{path}/results.csv', index=False)

# %%
