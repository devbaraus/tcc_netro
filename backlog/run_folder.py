from operator import itemgetter
from itertools import groupby
import os
import argparse

parser = argparse.ArgumentParser(description='Arguments algo')

parser.add_argument('-f', type=str,
                    action='store',
                    dest='folder',
                    required=False,
                    help='Parent Path',
                    default='')

args = parser.parse_args()

all_files = []

for root, dirs, files in os.walk(args.folder):
    for file in files:
        relativePath = os.path.relpath(root, args.folder)
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
    dirs.append(folder)

    # for file in files:
    #     print('File:', file[2])

print(' '.join(dirs))
