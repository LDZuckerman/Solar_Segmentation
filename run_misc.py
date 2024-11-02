import argparse
import run_WNet
import numpy as np
import os, sys, shutil
import pickle, json
import astropy.io.fits as fits
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import skimage
import os, sys, shutil
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
import astropy.io.fits as fits
from utils import run_utils, data_utils, models

parser = argparse.ArgumentParser()
parser.add_argument("-task", "--task", type=str, required=True)
parser.add_argument("-flag", "--flag", type=str, required=True)
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
    
if args.task == 'prep_trad_data':
    '''
    Save all (original) MURaM data and MURaM mag data images as single array for easy access when training simple models
    '''
    
    # Compile all normed imags and mag imgs into arrays of [n_imgs, N, N]
    data = []
    true = []
    for file in os.listdir('../Data/MURaM/'): 
        if 'Norm' in file:
            dat = fits.open('../Data/MURaM/'+file)[0].data
            mag = fits.open('../Data/MURaM_mag/'+file.replace('Norm_I_out','tau_slice_1.000').replace('.fits','_Bz.fits'))[0].data
            data.append(dat) #data = np.concatenate((data, dat), axis=1)
            true.append(mag) #true = np.concatenate((true, mag), axis=1)
    data = np.array(data)
    true = np.array(true)
    print(f'{data.shape[0]} imgs of shape {(data.shape[1], data.shape[2])}')
    
    if args.flag == 'seperate_test_imgs': # dont concatenate test obs (still flatten, but return list of obs)
    
        # Seperate train and test sets
        test_idxs = np.random.choice(np.linspace(0, data.shape[0]-1, data.shape[0], dtype=int), int(data.shape[0]*0.4), replace=False)
        print(f'{data.shape[0]-len(test_idxs)} for train, {len(test_idxs)} imgs for test')
        data_train = []; true_train = []
        data_test = []; true_test = []
        count = 0
        for i in range(data.shape[0]):
            if i in test_idxs:
                data_test.append(data[i]); true_test.append(true[i])
                count += 1
            else:
                data_train.append(data[i]); true_train.append(true[i])
        data_train = np.array(data_train); true_train = np.array(true_train)
        data_test = np.array(data_test); true_test = np.array(true_test)
        print(f'{len(data_train)} train obs and {len(data_test)} test obs')
           
        # Prepare all train data into single DF
        print(f'pre-processing full set train from shape {data_train.shape} to DFs of flattened, concatenated, values')
        X_train, Y_train = data_utils.preproccess(data, true) # X is DF where each col is (some transform of) flattened and concatenated images
        print(f'Saving train X of length {len(X_train)}')
        pickle.dump([X_train, Y_train],open('../Data/tradData_MURaM_mag_train.pkl','wb'))
        
        # Prepare seperate DFs for each test obs
        print(f'pre-processing full set test into {data_test.shape[0]} DFs')
        X_test_list = []
        Y_test_list = []
        for i in range(data_test.shape[0]):
            X_test, Y_test = data_utils.preproccess(data_test[i], true_test[i]) # X is DF where each col is (some transform of) single flattened images
            X_test_list.append(X_test)
            Y_test_list.append(Y_test)
        print(f'Saving test X list of length {len(X_test_list)}')
        pickle.dump([X_test_list, Y_test_list], open('../Data/tradData_MURaM_mag_test.pkl','wb'))
        
        # Prepare subset (1/4) of train data into single DF so can load in NB 
        train_sub_idxs = np.random.choice(np.linspace(0, len(data_train)-1, len(data_train), dtype=int), int(len(data_train)/4), replace=False)
        data_train_sub = []; true_train_sub = []
        for idx in train_sub_idxs:
            data_train_sub.append(data_train[idx])
            true_train_sub.append(true_train[idx])
        data_train_sub = np.array(data_train_sub)
        true_train_sub = np.array(true_train_sub)
        print(f'Creating subset of train data containing {len(data_train_sub)}/4 = {len(data_train_sub)} imgs')
        print(f'pre-processing subset train from shape {data_train_sub.shape} to DFs of flattened, concatenated, values')
        X_train_sub, Y_train_sub = data_utils.preproccess(data_train_sub, true_train_sub) # X is DF where each col is (some transform of) flattened and concatenated images
        print(f'Saving train X subset of length {len(X_train_sub)}')
        pickle.dump([X_train_sub, Y_train_sub],open('../Data/tradData_MURaM_mag_train_subset.pkl','wb'))
        
        # Prepare seperate DFs for each test obs in subset
        test_sub_idxs = np.random.choice(np.linspace(0, len(data_test)-1, len(data_test), dtype=int), int(len(data_test)/4), replace=False)
        data_test_sub = []; true_test_sub = []
        for idx in test_sub_idxs:
            data_test_sub.append(data_test[idx])
            true_test_sub.append(true_test[idx])
        data_test_sub = np.array(data_test_sub)
        true_test_sub = np.array(true_test_sub)
        print(f'Creating subset of test data containing {len(data_test)}/4 = {len(data_test_sub)} imgs')
        print(f'pre-processing full set test data into {data_test_sub.shape[0]} DFs')
        X_test_sub_list = []
        Y_test_sub_list = []
        for i in range(data_test_sub.shape[0]):
            X_test_sub, Y_test_sub = data_utils.preproccess(data_test_sub[i], true_test_sub[i]) # X is DF where each col is (some transform of) single flattened images
            X_test_sub_list.append(X_test_sub)
            Y_test_sub_list.append(Y_test_sub)
        print(f'Saving test X subset of length {len(X_test_sub)}')
        pickle.dump([X_test_sub_list, Y_test_sub_list],open('../Data/tradData_MURaM_mag_test_subset.pkl','wb'))
        
             
    else: # Treat test set the same, e.g. X as just 1D list of pixel values 

        # Prepare data
        print(f'pre-processing full set data and trues from shape {data.shape} to DFs of flattened, concatenated, values')
        X, Y = data_utils.pre_proccess(data, true) # all images flattened and concatenated
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=20)
        pickle.dump([X_train, X_test, Y_train, Y_test],open('../Data/tradData_MURaM_mag.pkl','wb'))

        # Prepare subset (1/4) so can load in NB (split along 'short' axis to encompass all)
        data_sub = data[0:int(np.shape(data)[0]/4), :]
        true_sub = true[0:int(np.shape(data)[0]/4), :]
        print(f'pre-processing subset data and trues from shape {data_sub.shape} to DFs of flattened, concatenated, values')
        X_sub, Y_sub = data_utils.pre_proccess(data_sub, true_sub) # all images flattened and concatenated
        X_train_sub, X_test_sub, Y_train_sub, Y_test_sub = train_test_split(X_sub, Y_sub, test_size=0.4, random_state=20)
        pickle.dump([X_train_sub, X_test_sub, Y_train_sub, Y_test_sub],open('../Data/tradData_MURaM_mag_subset.pkl','wb'))
        
    
if args.task == 'debug_fullnorm_run':
    
    # Analogous models 
    exp_dict = json.load(open('../model_runs_seg/MURaM/WNetM_1E/exp_file.json'))
    exp_dict_fn = json.load(open('../model_runs_seg/MURaM/WNetMfn_1Eb/exp_file.json'))
    #print(f'image norm exp_file\n{exp_dict}\nfull norm exp_file\n{exp_dict_fn}')

    for d in [exp_dict, exp_dict_fn]:
        print(f'{d["WNet_name"]}', flush=True)
        channels = d['channels'] 
        exp_outdir = f'../model_runs_seg/{d["sub_dir"]}/{d["WNet_name"]}/'
        train_ds = data_utils.dataset(image_dir=d['img_dir'], mask_dir=d['seg_dir'], set='train', norm=False, n_classes=d['n_classes'], channels=channels, randomSharp=d['randomSharp'], im_size=d['img_size'], add_mag=False, inject_brightmasks=False) # multichannel=True, channels=['deltaBinImg'], 
        train_loader = DataLoader(train_ds, batch_size=d['batch_size'], pin_memory=True, shuffle=True)
        in_channels = len(channels); out_channels = in_channels
        model = models.WNet(squeeze=d['n_classes'], ch_mul=64, in_chans=in_channels, out_chans=out_channels, kernel_size=d['kernel_size'], padding_mode=d['padding_mode'], reconstruct_mag=False, pretrained_dec="None").to(torch.device('cpu'))
        optimizer = torch.optim.SGD(model.parameters(), lr=d["learning_rate"])
        for epoch in range(1): #d['num_epochs']):
            print(f'epoch {epoch}', flush=True)
            #enc_losses, rec_losses = run_utils.train_WNet(train_loader, model, optimizer, k=d['n_classes'], img_size=(d['img_size'], d['img_size']), exp_outdir=exp_outdir, WNet_id=WNet_id, smooth_wght=d['smooth_wght'], blob_wght=d['blob_wght'], ncut_wght=d['ncut_wght'], epoch=epoch,  device=device, target_pos=0, weights=d['weights'], reconstruct_mag=d['reconstruct_mag'], save_examples=False, temp_dir=exp_outdir)      
            for (idx, batch) in enumerate(train_loader):
                    X = batch[0] # batch is [images, labels]
                    #wnet, enc_loss, rec_loss, enc, dec = train_wnet_batch(wnet, optimizer, X, k, img_size=img_size, smooth_wght=smooth_wght, ncut_wght=ncut_wght, blob_wght=blob_wght, device=device, freeze_dec=freeze_dec, target_pos=target_pos, weights=weights, debug=debug, reconstruct_mag=eval(str(reconstruct_mag)), epoch=epoch, temp_dir=temp_dir)
                    softmax = nn.Softmax2d(); soft_n_cut_loss = run_utils.get_loss_func('soft_n_cut_loss'); smoothLoss = run_utils.get_loss_func('OpeningLoss2D'); blobloss = run_utils.get_loss_func('cohesion_loss'); multichannel_MSE_loss = run_utils.get_loss_func('multichannel_MSE_loss')
                    model_input = X.to('cpu')
                    enc = model(model_input, compute_dec=False) # predict seg of k classes 
                    n_cut_loss = soft_n_cut_loss(X, softmax(enc),  img_size=(d['img_size'], d['img_size'])) # if reconstruct_mag, include mag channel in assesment of intra-cluster similarity
                    enc_loss = d['ncut_wght']*n_cut_loss + d['blob_wght']*blobloss(enc, temp_dir=exp_outdir) + d['smooth_wght']*smoothLoss(softmax(enc))
                    enc_loss.backward() 
                    optimizer.step()
                    optimizer.zero_grad()
                    rec = model(model_input, compute_dec=True) # predict image (all channels)
                    rec_loss = multichannel_MSE_loss(X, rec,d['weights']) # if reconstruct_mag, include mag channel in assesment of reconstruction accuracy (rec will have n_channels + 1 layers)   # from reproduction (MSELoss betwn input and rec imag) BUT with added channel weights
                    rec_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    print(f'\t[batch {batch}, {enc_loss.detach()}, {rec_loss.detach()}] ', end='', flush=True)
                    if batch == 20: break
    
    print("DONE", flush=True)
    