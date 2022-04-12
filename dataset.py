import json
import os
import shutil
from joblib import Parallel, delayed

import librosa
import numpy as np
import pandas as pd
import scipy.io as sio
from praudio import utils
from sklearn.model_selection import train_test_split
from plot import plot_class_distribution

from utils import merge_dicts

DATASET_PATH = '../datasets/base_portuguese'
OUTPUTDIR_PATH = './dataset/base'
SAMPLES_TO_CONSIDER = 22050  # 1 sec. of audio
SEED = 42


def annotate_dataset(dataset_path: str, output_path: str, sr: int = 24000, extract=[], plot_distribution=False):
    """
    It takes in a dataset path and outputs a metadata.csv file and a folder of audio files.

    :param dataset_path: The path to the dataset folder
    :type dataset_path: str
    :param output_path: the path where the audio files will be saved
    :type output_path: str
    :param sample_rate: The sample rate of the audio files, defaults to 24000
    :type sample_rate: int (optional)
    """

    extract = [str(k) for k in extract]

    base_data = {
        "mapping": [],
        "label": [],
        "sample_rate": [],
        "length": [],
        "filename": [],
    }

    utils.create_dir_hierarchy(output_path + '/audio')

    # loop through all sub-dirs
    def _run(i, dirpath, filenames):
        # for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        data = base_data.copy()

        # ensure we're at sub-folder level
        if dirpath is not dataset_path:
            if len(extract) and not dirpath.split("/")[-1] in extract:
                return data

            label = dirpath.split("/")[-1]
            # save label (i.e., sub-folder name) in the mapping
            print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path,
                                                   sr=sr,
                                                   mono=True)

                data["mapping"].append(label)
                data['sample_rate'].append(sample_rate)
                data['length'].append(len(signal) / sample_rate)
                data["label"].append(i - 1)
                data["filename"].append(f)

                sio.wavfile.write(
                    f'{output_path}/audio/{f}', sample_rate, signal)

        return data

    dicts = Parallel(n_jobs=-1)(delayed(_run)(i, dirpath, filenames)
                                for i, (dirpath, _, filenames) in enumerate(os.walk(dataset_path)))

    data = merge_dicts(base_data, *dicts)

    if plot_class_distribution:
        labels, count = np.unique(data['label'], return_counts=True)

        plot_class_distribution(labels,
                                count, save_path=f'{output_path}')

    pd.DataFrame.from_dict(data).to_csv(f'{output_path}/metadata.csv',
                                        index=False)


def annotate_inference(dataset_path: str, output_path: str, model_path: str, catalog_path: str, extract=[], plot_distribution=False):
    """

    :param dataset_path: the path to the dataset folder
    :type dataset_path: str
    :param output_path: the path where the output files will be saved
    :type output_path: str
    :param model_path: the path to the model
    :type model_path: str
    :param catalog_path: The path to the catalog.csv file
    :type catalog_path: str
    :param sr: The sample rate of the audio files, defaults to 24000
    :type sr: int (optional)
    """
    extract = [str(k) for k in extract]

    base_data = {
        "mapping": [],
        "label": [],
        "sample_rate": [],
        "length": [],
        "filename": [],
    }

    shutil.rmtree(output_path, ignore_errors=True)

    overview = json.load(open(f'{model_path}/overview.json'))

    model_mapping = overview['classes']

    df_catalog = pd.read_csv(catalog_path)

    inf_mapping = {}

    for _, row in df_catalog.iterrows():
        if row['inference'] > 0:
            inf_mapping[row['inference']] = model_mapping[str(row['train'])]

    people = []

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(f'{dataset_path}/data/')):
        for f in filenames:
            file_path = os.path.join(dirpath, f)

            file_json = json.load(open(file_path))

            if file_json['contrib'] == 'true' or file_json['contrib'] == True:
                people.append(str(file_json['_id']))

    utils.create_dir_hierarchy(output_path + '/audio')

    # loop through all sub-dirs
    # for i, (dirpath, _, filenames) in enumerate(os.walk(f'{dataset_path}/audio/')):
    def _run(i, dirpath, filenames):
        data = base_data.copy()

        # ensure we're at sub-folder level
        inf_keys = [str(k) for k in inf_mapping.keys()]

        if dirpath is not dataset_path and dirpath.split("/")[-1] in people and dirpath.split("/")[-1] in inf_keys:
            if len(extract) and not dirpath.split("/")[-1] in extract:
                return data

            # save label (i.e., sub-folder name) in the mapping
            mapping = int(dirpath.split("/")[-1])
            label = inf_mapping[mapping]

            print("\nProcessing: '{}'".format(mapping))

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path,
                                                   sr=overview['sample_rate'],
                                                   mono=True)

                data["mapping"].append(mapping)
                data['sample_rate'].append(sample_rate)
                data['length'].append(len(signal) / sample_rate)
                data["label"].append(label)
                data["filename"].append(f)

                sio.wavfile.write(
                    f'{output_path}/audio/{f}', sample_rate, signal)

        return data

    # exit()

    dicts = Parallel(n_jobs=-1)(delayed(_run)(i, dirpath, filenames)
                                for i, (dirpath, _, filenames) in enumerate(os.walk(f'{dataset_path}/audio/')))

    data = merge_dicts(base_data, *dicts)

    if plot_class_distribution:
        labels, count = np.unique(data['label'], return_counts=True)

        plot_class_distribution(labels,
                                count, save_path=f'{output_path}')

    pd.DataFrame.from_dict(data).to_csv(
        f'{output_path}/metadata.csv', index=False)


def split_dataset(input_dataset: str, output_path: str, validation: bool = True, plot_distribution=True):
    """
    It splits the dataset into train, test and validation subsets, and copies the audio files to the
    corresponding folders

    :param input_dataset: The path to the dataset folder
    :type input_dataset: str
    :param output_path: the path where the dataset will be created
    :type output_path: str
    :param validation: If True, the validation set will be created, defaults to True
    :type validation: bool (optional)
    """

    df = pd.read_csv(f'{input_dataset}/metadata.csv')

    labels = df['label'].tolist()

    subsets = {}

    X_train, X_test, X_valid, y_train, y_test, y_valid = [], [], [], [], [], []

    X_train, X_test, y_train, y_test = train_test_split(df,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=SEED,
                                                        stratify=labels)

    subsets['train'] = X_train
    subsets['test'] = X_test

    if validation:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                              y_train,
                                                              test_size=0.2,
                                                              random_state=SEED,
                                                              stratify=y_train)
        subsets['train'] = X_train
        subsets['valid'] = X_valid

    # check if all subsets have all labels
    if validation and not (list(set(y_valid)) == list(set(y_test)) == list(set(y_train))):
        raise BaseException('All subsets need to contain the same label')
    elif not (list(set(y_test)) == list(set(y_train))):
        raise BaseException('All subsets need to contain the same label')

    for key, value in subsets.items():

        shutil.rmtree(f'{output_path}/{key}', ignore_errors=True)

        utils.create_dir_hierarchy(f'{output_path}/{key}/audio')

        if plot_class_distribution:
            labels, count = np.unique(value['label'], return_counts=True)

            plot_class_distribution(labels,
                                    count, save_path=f'{output_path}/{key}')

        df_subset = pd.DataFrame.from_dict(value)
        df_subset.to_csv(f'{output_path}/{key}/metadata.csv', index=False)

        for audio in df_subset['filename'].tolist():
            src = f'{input_dataset}/audio/{audio}'
            dst = f'{output_path}/{key}/audio/{audio}'

            # if not os.path.exists(src):
            shutil.copy(src, dst)


if __name__ == '__main__':
    annotate_inference('/src/datasets/ifgaudio',
                       '/src/tcc/dataset/inference',
                       '/src/tcc/models/base_portuguese_40/SEG_1/MFCC_18/1649037587_71.875',
                       '/src/tcc/catalog.csv')
    # prepare_raw_dataset(DATASET_PATH, OUTPUTDIR_PATH)
    # split_dataset('/src/tcc_devbaraus/dataset/base', '/src/tcc_devbaraus/dataset', True)
