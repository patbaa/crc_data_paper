################################################################################
# imports

import os
import argparse
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

################################################################################
# command-line arguments 

parser = argparse.ArgumentParser()

parser.add_argument('--zoom_level', type=int, required=True)

parser.add_argument('--out_jpg_folder', type=Path, required=True)

parser.add_argument('--out_meta_folder', type=Path, required=True)
                    
parser.add_argument('--n_threads', type=int, default=8)

parser.add_argument('--data_path', type=Path, required=True)

args = parser.parse_args()

args.out_jpg_folder.mkdir(parents=True, exist_ok=True)
args.out_meta_folder.mkdir(parents=True, exist_ok=True)
################################################################################


slideIDs = [str(i).zfill(3) for i in range(1, 201)]
mrxs_files = [args.data_path.as_posix() + '/slides/' + i + '.mrxs' for i in slideIDs]
print(f'{len(mrxs_files)} files will be processed')

# full path to the mrxs file
def call_cutout(mrxs):
    fname   = Path(mrxs).stem
    masks = args.data_path.as_posix() + '/qupath_project/masks/'
    
    os.system((f'python3 mrxs_patcher.py --crop_size {512} '
               f'--out_meta {args.out_meta_folder.as_posix()} '
               f'--out_jpg {args.out_jpg_folder.as_posix() + "/" + fname + "/"} '
               f'--in_mask_dir {masks} '
               f'--in_mrxs {mrxs} '
               f'--zoom_level {args.zoom_level}'))

with Pool(args.n_threads) as p:
    p.map(call_cutout, mrxs_files)
