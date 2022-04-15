# %%
import math
import os
import shutil
from unittest import signals
import numpy as np
import audiomentations as am
import matplotlib.pyplot as plt
import librosa
from praudio import utils
import noisereduce as nr
from random import random

from joblib import Parallel, delayed

import os
import audiomentations as am
import pandas as pd
import librosa
from praudio import utils
import scipy.io as sio
import noisereduce as nr
from sklearn.utils import shuffle
from utils import arr_dimen

from utils import merge_dicts
from plot import plot_class_distribution


def augment_signal(samples: list, augment_size: int = 1):
    """
    It takes a list of samples and returns a new list of samples that is the same as the original list,
    but repeated a number of times

    :param samples: list
    :type samples: list
    :param augment_size: how many times to repeat the signal, defaults to 1
    :type augment_size: int (optional)
    :return: the samples repeated along the axis.
    """

    if isinstance(samples, (list)):
        samples = np.array(samples)

    return np.tile(samples, (augment_size, 1))


def transform_samples(samples: list, sample_rate: int, transformations: list, reduce_noise: float = 0, shuffle: bool = False):
    """
    Given a signal, sample rate, a list of transformations, and an augment size,
    this function will return a list of augmented signals

    :param signal: The signal to be augmented
    :type signal: np.array
    :param sample_rate: int, the sample rate of the signal
    :type sample_rate: int
    :param transformations: list of transformations to be applied to the signal
    :type transformations: list
    :param augment_size: the number of augmented samples to generate
    :type augment_size: int
    :return: A list of augmented signals.
    """
    augment_composition = am.Compose(transformations, shuffle=shuffle)

    if len(samples.shape) == 1:
        if reduce_noise and random() < reduce_noise:
            samples = nr.reduce_noise(samples, sr=sample_rate)

        return augment_composition(samples, sample_rate)

    aug_samples = []

    for sample in samples:
        if reduce_noise and random() < reduce_noise:
            sample = nr.reduce_noise(sample, sr=sample_rate)

        augmented_sample = augment_composition(sample, sample_rate)

        aug_samples.append(augmented_sample)

    return np.array(aug_samples)


def represent_signal(signal: list, sample_rate: int, plot: bool = False, **mfcc_params):
    """
    Computes the MFCCs of a signal

    :param signal: The input signal from which to compute features. Should be an N*1 array
    :type signal: np.array
    :param sample_rate: The sample rate of the audio file
    :type sample_rate: int
    :param plot: If True, plots the MFCC as an image, defaults to False
    :type plot: bool (optional)
    :return: The MFCCs of the signal.
    """

    mfcc = []

    def _plot(x_label='Frame Index', y_label='Index', cmap='magma', size=(10, 6)):
        if size:
            plt.figure(figsize=(10, 6), frameon=True)

        plt.imshow(mfcc,
                   origin='lower',
                   aspect='auto',
                   cmap=cmap,
                   interpolation='nearest')

        plt.title('MFCC')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()
        plt.show()
        plt.savefig(f'/src/tcc/plot/mfcc.png', dpi=300)
        plt.close()

    mfcc = librosa.feature.mfcc(signal,
                                sr=sample_rate,
                                **mfcc_params)

    if plot:
        _plot()

    return mfcc


def segment_signal(signal: list, sample_rate: int, segment_length: int, overlap_size: float = 0, plot=False):
    """
    It takes a signal and splits it into segments

    :param signal: the audio signal to be segmented
    :type signal: np.array
    :param sample_rate: The sample rate of the audio file
    :type sample_rate: int
    :param segment_length: the length of the segments in seconds
    :type segment_length: int
    :param overlap_size: The amount of overlap between segments, defaults to 0
    :type overlap_size: float (optional)
    :param plot: If True, the segmented audio will be plotted, defaults to False (optional)
    :return: A list of segments.
    """
    signal = np.array(signal)

    segments = []
    seg_positon = []

    def _plot_segments():
        shutil.rmtree('/src/tcc/plot', ignore_errors=True)
        os.mkdir('/src/tcc/plot')

        duration = signal.size / sample_rate
        time = np.arange(0, duration, 1/sample_rate)

        for i in range(len(segments)):
            fake_signal = np.zeros(len(signal))
            fake_signal[seg_positon[i][0]:seg_positon[i][1]] = segments[i]

            plt.plot(time, signal)
            plt.plot(time, fake_signal)

            plt.margins(x=0)
            plt.ylim(1, -1)
            plt.show()
            plt.savefig(f'/src/tcc/plot/segment_{i}.png', dpi=300)
            plt.close()

    overlap = 1 - overlap_size
    size_segment = sample_rate * segment_length
    size_overlap_segment = size_segment * overlap
    qtd_segments = 0
    flag = 1
    start_segment = 0

    while flag == 1:
        if start_segment + size_segment > signal.size:
            flag = 0
        else:
            qtd_segments = qtd_segments + 1
            start_segment = start_segment + size_overlap_segment

    for i in list(range(0, qtd_segments)):
        start_seg = int(i * sample_rate * segment_length * overlap)
        end_seg = int(start_seg + sample_rate * segment_length)

        segment_audio = signal[start_seg:end_seg]

        seg_positon.append([start_seg, end_seg])
        segments.append(segment_audio)

    if plot:
        _plot_segments()

    return np.array(segments)


def segment_dataset(input_dir: str,
                    output_dir: str,
                    base_trans: list = [],
                    extra_trans: list = [],
                    augment_size: int = 0,
                    overlap_size: float = 0.0,
                    segment_length: int = 1,
                    plot_distribution: bool = False,
                    aug_per_segment: bool = False,
                    reduce_noise: bool = False):

    df = pd.read_csv(f'{input_dir}/metadata.csv')

    base_dict = {
        "label": [],
        "sample_rate": [],
        "length": [],
        "artists": [],
        "album": [],
        "title": [],
        "genre": [],
        "thumbnail": [],
        "filename": [],
        "aug_filename": [],
        "transformations": [],
    }

    def _run(i: int):
        data = {**base_dict}

        row = df.iloc[i, :]

        src_filename = utils.remove_extension_from_file(row['filename'])

        signal, sample_rate = librosa.load(
            f'{input_dir}/audio/{row["filename"]}', sr=row["sample_rate"], mono=True)

        ### save original segments ###
        transformations_name = '-'.join([
            trans.__class__.__name__ for trans in base_trans])

        if aug_per_segment:
            segments = segment_signal(signal,
                                      sample_rate,
                                      segment_length=segment_length,
                                      overlap_size=overlap_size,
                                      plot=False)

            segments = transform_samples(segments,
                                         sample_rate,
                                         base_trans,
                                         reduce_noise,
                                         shuffle=True)

        else:
            augmented_signal = transform_samples(signal,
                                                 sample_rate,
                                                 base_trans,
                                                 1)

            segments = segment_signal(augmented_signal,
                                      sample_rate,
                                      segment_length=segment_length,
                                      overlap_size=overlap_size,
                                      plot=False)

        for indexI, segment in enumerate(segments):

            data['label'].append(row['label'])
            data['sample_rate'].append(row['sample_rate'])
            data['length'].append(segment.size/sample_rate)

            seg_filename = f'{src_filename}_{indexI}.wav'

            data['artists'].append(row['artists'])
            data['album'].append(row['album'])
            data['title'].append(row['title'])
            data['genre'].append(row['genre'])
            data['thumbnail'].append(row['thumbnail'])
            data['filename'].append(row['filename'])
            data['aug_filename'].append(seg_filename)

            sio.wavfile.write(f'{output_dir}/audio/{seg_filename}',
                              sample_rate,
                              segment)

            data['transformations'].append(transformations_name)

        ### save augmented segments ###
        if augment_size <= 0:
            return data

        transformations = [*base_trans, *extra_trans]

        transformations_name = '-'.join([
            trans.__class__.__name__ for trans in transformations])

        augmented_signal = augment_signal(signal, augment_size=augment_size)

        transformed_signals = transform_samples(augmented_signal,
                                                sample_rate,
                                                transformations)

        for indexI, transformed_signal in enumerate(transformed_signals):

            segments = segment_signal(transformed_signal,
                                      sample_rate,
                                      segment_length,
                                      overlap_size,
                                      plot=False)

            for indexJ, segment in enumerate(segments):

                data['label'].append(row['label'])
                data['sample_rate'].append(row['sample_rate'])
                data['length'].append(segment.size/sample_rate)

                aug_filename = f'{src_filename}_{indexI}_{indexJ}.wav'

                data['artists'].append(row['artists'])
                data['album'].append(row['album'])
                data['title'].append(row['title'])
                data['genre'].append(row['genre'])
                data['thumbnail'].append(row['thumbnail'])
                data['filename'].append(row['filename'])
                data['aug_filename'].append(aug_filename)

                sio.wavfile.write(f'{output_dir}/audio/{aug_filename}',
                                  sample_rate,
                                  segment)

                data['transformations'].append(transformations_name)

        return data

    utils.create_dir_hierarchy(f'{output_dir}/audio')

    # dicts = [_run(i) for i in range(len(df))]

    dicts = Parallel(n_jobs=-1)(delayed(_run)(i) for i in range(len(df)))

    df_dict = merge_dicts(base_dict, *dicts)

    if plot_distribution:
        labels, count = np.unique(df_dict['label'], return_counts=True)

        plot_class_distribution([str(x) for x in labels],
                                count, save_path=f'{output_dir}')

    pd.DataFrame.from_dict(df_dict).to_csv(f'{output_dir}/metadata.csv',
                                           index=False)


def represent_dataset(input_dir, output_dir, **mfcc_params):
    df = pd.read_csv(f'{input_dir}/metadata.csv')

    mat_dict = df.to_dict(orient='list')

    mat_dict['representation'] = []

    # for i in [2]:
    # for i in range(len(df)):
    def _run(i: int):
        row = df.iloc[i, :]

        signal, sample_rate = librosa.load(
            f'{input_dir}/audio/{row["aug_filename"]}',
            sr=row["sample_rate"],
            mono=True)

        representation = represent_signal(signal,
                                          sample_rate,
                                          plot=False,
                                          **mfcc_params)

        return representation

    utils.create_dir_hierarchy(f'{output_dir}')

    representations = Parallel(n_jobs=-1)(delayed(_run)(i)
                                          for i in range(len(df)))

    mat_dict['representation'] = representations
    sio.savemat(f'{output_dir}/representation.mat', mat_dict)

    return mat_dict


def pipeline_signal(file_path: str = '', sample_rate: int = 24000, segment_length: int = 1, overlap_size: float = 0, transformations: list = [], **mfcc_params):
    signal, rate = librosa.load(file_path,
                                sr=sample_rate,
                                mono=True)

    augment_array = transform_samples([signal], rate, transformations, 1)

    segments_array = []

    representation_array = []

    for audio in augment_array:
        segments = segment_signal(audio,
                                  rate,
                                  segment_length,
                                  overlap_size)

        segments_array.extend(segments)

    for segment in segments_array:
        audio_rep = represent_signal(segment,
                                     sample_rate,
                                     plot=False,
                                     **mfcc_params)

        representation_array.append(audio_rep)

    return np.array(representation_array),  np.array(segments_array), np.array(augment_array)


# %%
if __name__ == '__main__':
    BASE_TRANSFORM = [
        am.Trim(top_db=20, p=1),
        am.Normalize(p=1),
    ]

    TRAIN_TRANSFORM = [
        am.AddGaussianSNR(min_snr_in_db=24, max_snr_in_db=40, p=0.8),
        am.HighPassFilter(min_cutoff_freq=60, max_cutoff_freq=100, p=0.8),
        am.LowPassFilter(min_cutoff_freq=3400, max_cutoff_freq=4000, p=0.8),
        am.TimeStretch(min_rate=0.75, max_rate=2,
                       leave_length_unchanged=False, p=0.5),
    ]

    TEST_VALID_TRANSFORM = [
        am.AddGaussianSNR(min_snr_in_db=24, max_snr_in_db=40, p=0.5),
        am.HighPassFilter(min_cutoff_freq=60, max_cutoff_freq=100, p=0.5),
        am.LowPassFilter(min_cutoff_freq=3400, max_cutoff_freq=4000, p=0.5),
        am.TimeStretch(min_rate=0.75, max_rate=2,
                       leave_length_unchanged=False, p=0.5),
    ]

    segment_dataset(f'/src/tcc_netro/dataset/spotify_20',
                    f'/src/tcc_netro/dataset/spotify_20/testing',
                    base_trans=[*BASE_TRANSFORM, *TEST_VALID_TRANSFORM],
                    overlap_size=0,
                    segment_length=3,
                    plot_distribution=True,
                    aug_per_segment=True)
