# %%
from operator import itemgetter
from itertools import groupby
import os
import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Arguments algo')

parser.add_argument('-i', type=str, action='store', dest='input', required=False, help='Dataset',
                    default=None)


args, _ = parser.parse_known_args()
# %%
all_files = []
path = args.input or '/src/tcc_netro/models/spotify_20'

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
    # if folder.find("inference") != -1:
    dirs.append(folder)

    # for file in files:
    #     print('File:', file[2])

# print(' '.join(dirs))

# %%
df_dict = {
    # 'classes': [],
    'segment_length': [],
    'sample_rate': [],
    'overlap_size': [],
    'augment_size': [],
    'n_mfcc': [],
    # 'inf_micro': [],
    # 'inf_macro': [],
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

    # df_dict['classes'].append(len(overview['classes']))
    df_dict['sample_rate'].append(overview['sample_rate'])
    df_dict['segment_length'].append(overview['segment_length'])
    df_dict['overlap_size'].append(overview['overlap_size'])
    df_dict['augment_size'].append(overview['augment_size'])
    df_dict['n_mfcc'].append(overview['representation']['n_mfcc'])
    # df_dict['inf_micro'].append(format_float(overview['f1_micro']))
    # df_dict['inf_macro'].append(format_float(overview['f1_macro']))
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
