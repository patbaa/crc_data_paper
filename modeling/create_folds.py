import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--zoom_level', type=int, required=True)
parser.add_argument('--csv_folder', type=Path, required=True)

args = parser.parse_args()


csv_files = list(Path(args.csv_folder).glob('*csv'))

meta = pd.DataFrame()

for i in csv_files:
    tmp = pd.read_csv(i.as_posix())
    tmp['slideID'] = [k.split('/')[-2] for k in tmp.fname.values]
    meta = meta.append(tmp)

slides = list(pd.unique(meta.slideID))
slides.sort()

print(f'{len(meta)} patches are collected in {len(slides)} unique slides.'.center(80, '='))

fold1 = list(slides[:40])
fold2 = list(slides[40:80])
fold3 = list(slides[80:120])
fold4 = list(slides[120:160])
fold5 = list(slides[160:200])

assert len(fold1) == len(fold2) == len(fold3) == len(fold4) == len(fold5)

meta[meta.slideID.isin(fold1)].to_csv(f'zoom{args.zoom_level}/fold1_test.csv', index=False)
meta[meta.slideID.isin(fold2)].to_csv(f'zoom{args.zoom_level}/fold2_test.csv', index=False)
meta[meta.slideID.isin(fold3)].to_csv(f'zoom{args.zoom_level}/fold3_test.csv', index=False)
meta[meta.slideID.isin(fold4)].to_csv(f'zoom{args.zoom_level}/fold4_test.csv', index=False)
meta[meta.slideID.isin(fold5)].to_csv(f'zoom{args.zoom_level}/fold5_test.csv', index=False)

meta[meta.slideID.isin(fold2 + fold3 + fold4 + fold5)].to_csv(f'zoom{args.zoom_level}/fold1_train.csv', index=False)
meta[meta.slideID.isin(fold1 + fold3 + fold4 + fold5)].to_csv(f'zoom{args.zoom_level}/fold2_train.csv', index=False)
meta[meta.slideID.isin(fold1 + fold2 + fold4 + fold5)].to_csv(f'zoom{args.zoom_level}/fold3_train.csv', index=False)
meta[meta.slideID.isin(fold1 + fold2 + fold3 + fold5)].to_csv(f'zoom{args.zoom_level}/fold4_train.csv', index=False)
meta[meta.slideID.isin(fold1 + fold2 + fold3 + fold4)].to_csv(f'zoom{args.zoom_level}/fold5_train.csv', index=False)

print(f'{len(meta[meta.slideID.isin(fold1)])} patches in fold1')
print(f'{len(meta[meta.slideID.isin(fold2)])} patches in fold2')
print(f'{len(meta[meta.slideID.isin(fold3)])} patches in fold3')
print(f'{len(meta[meta.slideID.isin(fold4)])} patches in fold4')
print(f'{len(meta[meta.slideID.isin(fold5)])} patches in fold5')
