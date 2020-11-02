import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

################################################################################
# command-line arguments 

parser = argparse.ArgumentParser()

parser.add_argument('--meta_csv', type=Path, 
                    default='/home/pataki/patho_scientificdata/global_metadata.csv')
parser.add_argument('--csv_path', type=Path, 
                    default='/home/pataki/patho_scientificdata/patched_data/')

args = parser.parse_args()

################################################################################

meta = pd.read_csv(args.meta_csv.as_posix(), dtype={'slideID':str})

mask_to_colors = {'lowgrade_dysplasia':     'blue',
                  'highgrade_dysplasia':    'orange',
                  'adenocarcinoma':         'green',
                  'suspicious_for_invasion':'red',
                  'inflammation':           'purple',
                  'resection_edge':         'brown',
                  'tumor_necrosis':         'pink',
                  'lymphovascular_invasion':'gray',
                  'artifact':               'olive'}

exported = list(args.csv_path.glob('*csv'))

def parse_mask(maskfile):
    coords = maskfile.replace(')-mask', '').split('_(')[-1].split(',')
     
    return {'topleft_x': int(coords[1]),
            'topleft_y': int(coords[2]),
            'width':     int(coords[3]),
            'height':    int(coords[4])}

def get_screenshot(fname, border=1):
    scale = 0.01
    annot = Path(fname).parent.glob(f'{Path(fname).stem.replace("shot_", "")}_annot*')
    coords = parse_mask(list(annot)[0].stem)
    data = plt.imread(fname)
    
    x_from = max(int(scale*coords['topleft_y']-border), 0)
    x_to   = int(scale*(coords['topleft_y']+coords['height'])+border)
    y_from = max(int(scale*coords['topleft_x']-border), 0)
    y_to   = int(scale*(coords['topleft_x']+coords['width'])+border)
    
    data = data[x_from:x_to, y_from:y_to, :]
    
    return data, {'xmin':coords['topleft_x'], 
                  'xmax':coords['topleft_x']+coords['width'], 
                  'ymin':coords['topleft_y'], 
                  'ymax':coords['topleft_y']+coords['height']}
                                    
# '/home/pataki/patho_scientificdata/patched_data/001_labels.csv'
def plot_wrapper(fname):
    ID = fname.stem.split('_')[0]
    screenshot_fname = f'/home/pataki/patho_scientificdata/mrxs_data/qupath_project/masks/shot_{ID}.jpg'
    
    #if Path(fname.parents[0].joinpath(f'overview_figures/{ID}.png')).exists():
    #    return 0
                                    
    screen_img, coords = get_screenshot(screenshot_fname)
    df = pd.read_csv(fname)                                
                                    
    plt.figure(figsize=(20, 10))
    plt.suptitle(ID, fontsize=25)
    ax = plt.subplot(121)
    ax.set_aspect(aspect=1)
    plt.scatter(df.topleft_x, df.topleft_y, c='k', label='normal')

    # plot the different conditions over each other
    # due to overlapping we might lose some information
    # but it is only for visualization and with transparent it look odd
    for i in mask_to_colors.keys():
        tmp_df = df[df[i] == 1]
        if len(tmp_df > 0):
            plt.scatter(tmp_df.topleft_x, tmp_df.topleft_y, c=mask_to_colors[i], label=i)

    plt.legend(fontsize=15)
    plt.axis('off')
    plt.xlim(coords['xmin'], coords['xmax'])
    plt.ylim(coords['ymin'], coords['ymax'])
    plt.gca().invert_yaxis()
    
    ax = plt.subplot(122)
    ax.set_aspect(aspect=1)
    plt.imshow(screen_img)
    plt.axis('off')
    
    
    #plt.show()
    plt.savefig(fname.parents[0].joinpath(f'overview_figures/{ID}.png'),
                bbox_inches = 'tight')
    plt.close()

exported[0].parents[0].joinpath('overview_figures').mkdir(parents=True, exist_ok=True)                    

for i in exported: plot_wrapper(i)                                    
