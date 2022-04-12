import os
import pandas as pd
import scipy.io as sio
import librosa


def load_mat_representation(input_path):
    """
    Load the mat file and return the data

    :param input_path: The path to the .mat file containing the data
    :return: A dictionary with the following keys:
        representation: The spectrogram of the audio file
        mapping: A mapping of the indices of the spectrogram to the corresponding labels
        label: The label of the audio file
        length: The length of the audio file
        filename: The filename of the audio file
        aug_filename: The filename of the augmented audio file
    """
    mat_dict = sio.loadmat(input_path)

    mapping = mat_dict['mapping'][0]
    label = mat_dict['label'][0]
    length = mat_dict['length'][0]
    filename = mat_dict['filename'][0]
    aug_filename = mat_dict['aug_filename']
    representation = mat_dict['representation']
    sample_rate = mat_dict['sample_rate']

    return {
        'representation': representation,
        'mapping': mapping,
        'label': label,
        'length': length,
        'filename': filename,
        'aug_filename': aug_filename,
        'sample_rate': sample_rate
    }


def load_metadata(input_dir):
    """
    Given a directory, load the metadata.csv file into a dictionary

    :param input_dir: The directory where the metadata.csv file is located
    :return: A dictionary. The keys are the column names of the metadata.csv file.
    """
    mat_dict = pd.read_csv(f'{input_dir}/metadata.csv').to_dict(orient='list')

    return mat_dict


def load_audios(input_dir):
    """
    Load all audio files in a directory and store them in a list

    :param input_dir: the directory where the audio files are located
    :return: A list of audio signals.
    """
    audios = []

    for i, (dirpath, _, filenames) in enumerate(os.walk(input_dir)):

        # ensure we're at sub-folder level
        if dirpath is not input_dir:

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file
                signal, _ = librosa.load(file_path)
                audios.append(signal)

    return audios


def load_all(input_dir):
    metadata = load_metadata(input_dir)
    audios = load_audios(input_dir)

    return {
        **metadata,
        'audio': audios
    }
