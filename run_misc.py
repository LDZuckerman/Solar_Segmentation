import argparse
import run_WNet
import numpy as np
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument("-task", "--task", type=str, required=True)
args = parser.parse_args()

if args.task == 're-test':

    ds =  [{"WNet_name": "WNet29nm",
            "n_classes": 3,
            "channels": ["X"],
            "img_dir": "../Data/UNetData_MURaM/norm_images/",
            "seg_dir": "../Data/UNetData_MURaM/seg_images/",
            "img_size": 128,
            "randomSharp": 'False',
            "padding_mode": "zeros",
            "batch_size": 16,},
           {"WNet_name": "WNet31nm",
            "n_classes": 3,
            "channels": ["X"],
            "img_dir": "../Data/UNetData_MURaM/norm_images/",
            "seg_dir": "../Data/UNetData_MURaM/seg_images/",
            "img_size": 128,
            "randomSharp": 'False',
            "batch_size": 16,
            "padding_mode": 'reflect'},
           {"WNet_name": "WNet34nm",
            "n_classes": 3,
            "channels": ["X","median_residual"],
            "img_dir": "../Data/UNetData_MURaM/norm_images/",
            "seg_dir": "../Data/UNetData_MURaM/seg_images/",
            "img_size": 128,
            "randomSharp": 'False',
            "batch_size": 16,
            "padding_mode": 'reflect'},
           {"WNet_name": "WNet35nm",
            "n_classes": 3,
            "channels": ["X","median_residual"],
            "img_dir": "../Data/UNetData_MURaM/norm_images/",
            "seg_dir": "../Data/UNetData_MURaM/seg_images/",
            "img_size": 128,
            "randomSharp": 'False',
            "batch_size": 16,
            "padding_mode": 'reflect'}]

    for d in ds:
        print(f'{d["WNet_name"]}')
        run_WNet.run_wnet_model(d, gpu=False, test_only=True)
    print('DONE')
        
if args.task == 'find_image':
    '''
    Go through MURaM test set to find image idxs corresponding to MURaM time-series test set image 58
    '''
    
    ts_im_58 = np.load('../WNet_runs/exp34nm/test_preds_MURaM/x_58.npy')[0] # should really not be sacing test set with each model but oh well
    all_ims = [f for f in os.listdir('../WNet_runs/exp29nm/test_preds_MURaM') if 'x_' in f]
    print(f'Searching through {len(all_ims)} images for match')
    for f in all_ims:
        im = np.load(f'../WNet_runs/exp29nm/test_preds_MURaM/{f}')[0]
        if np.all(im == ts_im_58):
            print(f'Image {f} seems to match')
    print('DONE')
    