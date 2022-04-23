# %%
import eyed3
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display

# %%
sr, signal = sio.wavfile.read(
    '/src/tcc_netro/dataset/spotify_120/SEG_5_OVERLAP_0_AUG_5/test/audio/18 Dollars - flora cash_0.wav')
_, signal2 = sio.wavfile.read(
    '/src/tcc_netro/dataset/spotify_120/audio/18 Dollars - flora cash.wav')


# %%
audiofile = eyed3.load(
    "/src/tcc_netro/dataset/spotify_20/mp3/Live That Long - Lewis Del Mar.mp3")

# %%
audiofile.tag.artist
# %%
audiofile.tag.album
# %%
audiofile.tag.genre.name
# %%
audiofile.tag.title
# %%
mat = sio.loadmat(
    '/src/tcc_netro/dataset/spotify_80/SEG_3_OVERLAP_0_AUG_5/MFCC_40/train/representation.mat')

mmean = np.mean(np.mean(mat['representation'], axis=1), axis=0)
mstd = np.std(np.std(mat['representation'], axis=1), axis=0)
plt.plot(mmean)
plt.plot(mstd)
plt.show()
plt.close()
# %%
mat = sio.loadmat(
    '/src/tcc_netro/dataset/spotify_80/SEG_3_OVERLAP_0_AUG_5/MFCC_40/test/representation.mat')

mmean = np.mean(np.mean(mat['representation'], axis=1), axis=0)
mstd = np.std(np.std(mat['representation'], axis=1), axis=0)
plt.plot(mmean)
plt.plot(mstd)
plt.show()
plt.close()
# %%
mat = sio.loadmat(
    '/src/tcc_netro/dataset/spotify_80/SEG_3_OVERLAP_0_AUG_5/MFCC_40/valid/representation.mat')

mmean = np.mean(np.mean(mat['representation'], axis=1), axis=0)
mstd = np.std(np.std(mat['representation'], axis=1), axis=0)
plt.plot(mmean)
plt.plot(mstd)
plt.show()
plt.close()

# %%
librosa.display.specshow(mat['representation'][3], x_axis='time')
plt.colorbar()
plt.tight_layout()
plt.title('mfcc')
plt.show()
plt.close()

# %%
df1 = pd.read_csv('/src/tcc_netro/dataset/spotify_80/metadata.csv')
df2 = pd.read_csv('/src/tcc_netro/dataset/spotify_60/metadata.csv')

# %%
df1_titles = df1['title'].to_list()
df2_titles = df2['title'].to_list()

df_comp = [d for d in df1_titles if d not in df2_titles]

print(df_comp.sort())
