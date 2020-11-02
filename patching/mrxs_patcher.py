################################################################################
# imports

import argparse
import openslide
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from collections import Counter

################################################################################
# command-line arguments 

parser = argparse.ArgumentParser()

parser.add_argument('--in_mrxs', type=Path, help='The .mrxs file to be processed', 
                    required=True)

parser.add_argument('--in_mask_dir', type=Path, help='The folder where the \
                    QuPath masks are stored', required=True)

parser.add_argument('--crop_size', type=int, help='The slide is patched to \
                    crop_size x crop_size smaller shunks', default=512)

parser.add_argument('--out_jpg', type=Path, help='Where the results are stored', 
                    required=True)

parser.add_argument('--out_meta', type=Path, help='Where the csv is stored', 
                    required=True)

parser.add_argument('--zoom_level', type=int, help='Zoom level', 
                    default=0)

args = parser.parse_args()

cs = args.crop_size

args.out_jpg.mkdir(parents=True, exist_ok=True)
args.out_meta.mkdir(parents=True, exist_ok=True)
################################################################################
# helper functions

valid_masks = ['dysplasia',
               'highgrade_dysplasia',
               'adenocarcinoma',
               'suspicious_for_invasion',
               'inflammation',
               'resection_edge',
               'tumor_necrosis',
               'lymphovascular_invasion',
               'artifact',
               'annotated']

# some annotators use slightly different names
possible_masks = valid_masks + ['necrosis', 'artefact']
                                
#example filename: 19_2524_annotated_(16.00,36845,114804,92637,70269)-mask
def parse_mask(maskfile, masktypes = possible_masks):
    coords = maskfile.replace(')-mask', '').split('_(')[-1].split(',')
    
    num_masks = 0
    for i in masktypes:
        if i in maskfile:
            mask_type = i
            num_masks += 1
    
    if num_masks == 2:
        if ('dysplasia' in maskfile) & ('highgrade_dysplasia' in maskfile):
            mask_type = 'highgrade_dysplasia'
            num_masks = 1
        elif ('tumor_necrosis' in maskfile) & ('necrosis' in maskfile):
            mask_type = 'tumor_necrosis'
            num_masks = 1
            
    if mask_type == 'artefact': mask_type = 'artifact'       
    if mask_type == 'necrosis': mask_type = 'tumor_necrosis'
        
    assert num_masks == 1, f'Invalid mask name occured: {maskfile}'
    
    return {'mask_type': mask_type,
            'scale':     float(coords[0]),
            'topleft_x': int(coords[1]),
            'topleft_y': int(coords[2]),
            'width':     int(coords[3]),
            'height':    int(coords[4])}


def load_masks(file_name, maskfiles, masktypes = possible_masks):
    # first solve the issue when one file's name is part of others
    # file1: ABC_I_dysplasia_(16.00,36845,114804,92637,70269)-mask
    # file2: ABC_II_dysplasia_(16.00,36845,114804,92637,70269)-mask
    valid_maskfiles = []
    for i in maskfiles:
        i = i.as_posix()
        assert i.count('(') == 1, (f'Mrxs files should not have "(" '
                                   f'in their name: {maskfile}')
        
        # keep file1: dysplasia_(16.00,36845,114804,92637,70269)-mask
        # keep file2: _dysplasia_(16.00,36845,114804,92637,70269)-mask
        mf = i.split('/' + file_name + '_')[-1]
        
        # keep file1: dysplasia
        # keep file2: _dysplasia
        if mf.split('_(')[0].strip().replace(' ', '_') in masktypes: 
            valid_maskfiles.append(Path(i))
        #strip is needed, someone named ' tumor_necrosis'
        #replace is needed, someone named 'tumor necrosis'
    
    masks = []
    for i in valid_maskfiles:
        mask_meta = parse_mask(i.stem)
        
        # annotated is special
        if mask_meta['mask_type'] == 'annotated':
            continue
            
        mask_img = Image.open(i)
        mask_img = mask_img.resize((int(mask_img.size[0]*mask_meta['scale']),
                                    int(mask_img.size[1]*mask_meta['scale'])))
        mask_img = np.array(mask_img)
        mask_img = np.swapaxes(mask_img, 0, 1)
        # converting to numpy a PIL image swaps the axes
        
        masks.append({'mask_type': mask_meta['mask_type'],
                      'mask_img':  mask_img,
                      'topleft_x': mask_meta['topleft_x'],
                      'topleft_y': mask_meta['topleft_y'],
                      'width':     mask_meta['width'],
                      'height':    mask_meta['height']})
        
    return masks

def contains_mask(img_topleft_x, img_topleft_y, mask, zoom_rate, 
                  img_width=cs, img_height=cs):
    tl_x = img_topleft_x - mask['topleft_x']
    tl_y = img_topleft_y - mask['topleft_y']
    
    
    cropped_mask = mask['mask_img'][max(0, tl_x):max(0, tl_x + img_width*zoom_rate), 
                                    max(0, tl_y):max(0, tl_y + img_height*zoom_rate)]
    
    if cropped_mask.sum() == 0:
        return False
    else:
        return cropped_mask.mean() > 0.5 # label only if >50% of pixels support

def get_crop(slide, topleft_x, topleft_y, crop_size=cs, zoom_level=args.zoom_level):
    crop = slide.read_region((topleft_x, topleft_y), zoom_level, (cs, cs))
    crop = np.array(crop)
    
    # make transparent pixels white
    crop[crop[...,-1] == 0] = np.array([255, 255, 255, 0], dtype=np.uint8)     
    
    return crop[:,:,:3] # return only RGB

# patch filtering logic
def keep_cropped_img(img):
    S = 1 - img.min(2)/(img.mean(2)+1e-8) # saturation
    I = img.mean(2) # intensity
    
    saturation_check = (S < 0.05).mean() <= 0.50
    intensity_check  = (I > 245).mean()  <= 0.50
    
    # keep images only when useless pixel's ratio is < 50%
    return (saturation_check & intensity_check)

def save_img(img, fname):
    img = Image.fromarray(img)
    img.save(fname, quality=80, subsampling=1)

################################################################################

slide = openslide.open_slide(args.in_mrxs.as_posix())

annot_mask = list(args.in_mask_dir.glob(f'{args.in_mrxs.stem}_annotated*'))
assert len(annot_mask) == 1, (f'{args.in_mrxs.as_posix()} has {len(annot_mask)} '
                              f'annotated masks. Should be exactly one!')
# ar = Annotated Region
ar = parse_mask(annot_mask[0].stem)

masks = load_masks(args.in_mrxs.stem, 
                   args.in_mask_dir.glob(f'{args.in_mrxs.stem}_*'))

meta_df = pd.DataFrame()

zoom_rate = [1, 2, 4, 8, 16, 32, 64, 128][args.zoom_level]
zoom_rate_check = slide.level_downsamples[args.zoom_level]
assert zoom_rate == zoom_rate_check

idx = 0        
for x in range(0, ar['width'] // cs, zoom_rate):
    for y in range(0, ar['height'] // cs, zoom_rate):
        
        crop = get_crop(slide, ar['topleft_x'] + x*cs, ar['topleft_y'] + y*cs)
        
        if keep_cropped_img(crop):
            fname = args.out_jpg.joinpath(str(idx) + '.jpg').as_posix()
            tmp_df = pd.DataFrame({'fname': [fname],
                                   'topleft_x': [ar['topleft_x'] + x*cs],
                                   'topleft_y': [ar['topleft_y'] + y*cs]})                    
            for m in valid_masks: tmp_df[m] = 0
               
            # we might have multiple annotations for the same patch                                                        
            for mask in masks:
                if contains_mask(ar['topleft_x'] + x*cs, 
                                 ar['topleft_y'] + y*cs, mask, zoom_rate = zoom_rate):
                    tmp_df[mask['mask_type']] = 1
            
            # save jpg image
            meta_df = meta_df.append(tmp_df)
            save_img(crop, fname)
            idx += 1

# patches where there is no annotations are normals
meta_df['normal'] = (meta_df[valid_masks].sum(1) == 0).astype(int)
meta_df['n_masks_for_slide'] = len(masks)
# during annotation lowgrade dysplasia was simply referred as dysplasia
meta_df['lowgrade_dysplasia'] = meta_df.dysplasia
meta_df.pop('dysplasia');
meta_df.drop('annotated', axis=1, inplace=True)
meta_df.to_csv(args.out_meta.joinpath(args.in_mrxs.stem + '_labels.csv'), 
               index=False)