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

parser.add_argument('--meta_file', type=Path)

parser.add_argument('--out_jpg_folder', type=Path, required=True)

parser.add_argument('--out_meta_folder', type=Path, required=True)
                    
parser.add_argument('--n_threads', type=int, default=8)

args = parser.parse_args()

args.out_jpg_folder.mkdir(parents=True, exist_ok=True)
args.out_meta_folder.mkdir(parents=True, exist_ok=True)
################################################################################


meta = pd.read_csv(args.meta_file, dtype={'slideID': object})
githubdir = '/home/pataki/patho_scientificdata/'
meta['slideID'] = githubdir + 'mrxs_data/img/' +  meta.slideID.values + '.mrxs'
print(f'{len(meta)} files will be processed')

# full path to the mrxs file
def call_cutout(mrxs):
    fname   = Path(mrxs).stem
    masks = githubdir + 'mrxs_data/qupath_project/masks/'
    
    os.system((f'python3 mrxs_patcher.py --crop_size {512} '
               f'--out_meta {args.out_meta_folder.as_posix()} '
               f'--out_jpg {args.out_jpg_folder.as_posix() + "/" + fname + "/"} '
               f'--in_mask_dir {masks} '
               f'--in_mrxs {mrxs} '
               f'--zoom_level {args.zoom_level}'))

with Pool(args.n_threads) as p:
    p.map(call_cutout, meta.slideID.values)
