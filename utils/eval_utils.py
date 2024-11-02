import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import pandas as pd
import json


########################################
# Helper functions for evaluating models
########################################

def onehot_to_map(y):
    '''
    Helper function to turn onehot predicted class layers into 2D array of truth labels (also changes tensor to numpy)
    '''
    # to deal with single image or batch
    if y.ndim == 4: 
        n_classes = y.shape[1]
        axis = 1
    else:
        n_classes = y.shape[0] 
        axis = 0 
        
    # if coming directly from GPU-trained model
    try:
        y = y.cpu().detach().numpy() 
    except AttributeError:
        y = y
    
    if n_classes == 2: # if binary 
        y = y[:,0,:,:] 
    elif n_classes == 3: # if 3 classs (y has 3 layers)
        y = np.argmax(y, axis=axis).astype(float) # turn 1-hot-encoded layers [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
        y[y == 2] = 1.5 # change from idx to class value 0 -> 0 (IG), 1 -> 1 (G), 2 -> 1.5 (BP)
    elif n_classes == 4: # if 4 classs (y has 3 layers)
        y = np.argmax(y, axis=axis) # turn 1-hot-encoded layers [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
        y = y/2 # change from idx to class value 0 -> 0 (IG), 1 -> 0.5 (DM), 2 -> 1 (G), 3 -> 1.5 (BP)

    return y

def map_to_onehot(y):
    '''
    Helper function to undo the above (for single image - no batch dim)
    '''
    vals = np.unique(y)
    pred = np.zeros((len(vals), np.shape(y)[0], np.shape(y)[1]))
    for i in range(len(vals)):
        pred[i,:,:][y==vals[i]] = 1
        
    return pred
    
    

def probs_to_preds(probs):
    '''
    Helper function to turn 3D class probs into 2D arrays of predictions (also changes tensor to numpy)
    '''
    if probs.shape[1] == 1: # if binary (predictions have 1 layer)
        preds = (probs > 0.5).float() # turn regression value (between zero and 1, e.g. "prob of being 1") into predicted 0s and 1s
    else: # if muliclasss (predictions have n_classes layers)
        preds = np.argmax(probs.detach().numpy(), axis=1) # turn probabilities [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]

    return preds

def get_netDF(modeldir='', tag=''):
    '''
    Helper function to create dataframe of all run models, their parameters, and results 
    '''
    expdirs = [f for f in os.listdir(modeldir) if os.path.isdir(f'{modeldir}/{f}') and tag in f]
    
    # Create DF from exp dicts
    all_info = []
    for expdir in expdirs:
        
        # Skip if not finished training 
        if not os.path.exists(f'{modeldir}/{expdir}/test_preds_MURaM'):
            print(f'Skipping {expdir}; not finished training')
            continue
        if not os.path.exists(f'{modeldir}/{expdir}/exp_file.json'):
            print(f'Skipping {expdir}; no exp_file found')
            continue
        exp_dict = json.load(open(f'{modeldir}/{expdir}/exp_file.json','rb'))
        
        # Add val mse
        rmse = prediction_validation_results(output_dir=f'{modeldir}/{expdir}/test_preds_MURaM')
        exp_dict['RMSE'] = rmse
       
        # Add data norm
        exp_dict['norm'] = 'full' if 'fullnorm' in exp_dict['img_dir'] else 'image' if 'norm' in exp_dict['img_dir'] else 'None'
    
        # Add to dict
        all_info.append(exp_dict)
    
    # Drop cols and sort 
    try:
        all_info = pd.DataFrame(all_info).drop(columns=['img_dir', 'true_dir', 'randomSharp','sub_dir'])
    except KeyError:
        all_info = pd.DataFrame(all_info).drop(columns=['img_dir', 'seg_dir', 'randomSharp','sub_dir'])
    all_info = all_info.sort_values(by='net_name')

    # Change col names for compact display
    all_info = all_info.rename(columns={"learning_rate":"lr","num_epochs":"ne"}) 
    
    return all_info 


def get_unetDF(modeldir='../UNet_runs/MURaM', tag=''):
    '''
    Helper function to create dataframe of all run models, thier parameters, and results 
    '''
    expdirs = [f for f in os.listdir(modeldir) if os.path.isdir(f'{modeldir}/{f}') and tag in f]
    
    # Create DF from exp dicts
    all_info = []
    for expdir in expdirs:
        
        # Skip if not finished training 
        if not os.path.exists(f'{modeldir}/{expdir}/test_preds_MURaM'):
            print(f'Skipping {expdir}; not finished training')
            continue
        if not os.path.exists(f'{modeldir}/{expdir}/exp_file.json'):
            print(f'Skipping {expdir}; no exp_file found')
            continue
        exp_dict = json.load(open(f'{modeldir}/{expdir}/exp_file.json','rb'))
        
        # Add val mse
        rmse = prediction_validation_results(output_dir=f'{modeldir}/{expdir}/test_preds_MURaM')
        exp_dict['RMSE'] = rmse
        
        # Add data norm
        exp_dict['norm'] = 'full' if 'fullnorm' in exp_dict['img_dir'] else 'image' if 'norm' in exp_dict['img_dir'] else 'None'
       
        # Add to dict
        all_info.append(exp_dict)
    
    # Drop cols and sort 
    all_info = pd.DataFrame(all_info).drop(columns=['img_dir', 'true_dir', 'img_size','randomSharp','task_dir'])
    if 'mag' in exp_dict['UNet_name']: 
        all_info =  all_info.drop(columns=['bp_wght'])
    all_info = all_info.sort_values(by='UNet_name')

    # Change col names for compact display
    all_info = all_info.rename(columns={"learning_rate":"lr","num_epochs":"ne"}) 
    
    return all_info 

def get_wnetDF(modeldir='../WNet_runs/MURaM', tag=''):
    '''
    Helper function to create dataframe of all run models and their parameters
    '''
    
    ignore = 'WNetM_1A_again'
    expdirs = [f for f in os.listdir(modeldir) if os.path.isdir(f'{modeldir}/{f}') and tag in f and ignore not in f]
    print(expdirs)
    
    # Create DF from exp dicts
    all_info = []
    for expdir in expdirs:
        
        # Skip if not finished training 
        if not os.path.exists(f'{modeldir}/{expdir}/test_preds_MURaM'):
            print(f'Skipping {expdir}; not finished training')
            continue
        if not os.path.exists(f'{modeldir}/{expdir}/exp_file.json'):
            print(f'Skipping {expdir}; no exp_file found')
            continue
        exp_dict = json.load(open(f'{modeldir}/{expdir}/exp_file.json','rb'))
        
        # Add kernel size and task dir for models run before this was added as a changable parameter
        if 'kernel_size' not in exp_dict.keys():
            exp_dict['kernel_size'] = 3
        if 'task_dir' not in exp_dict.keys():
            exp_dict['task_dir'] = 'MURaM'
       
        # Add data norm
        exp_dict['norm'] = 'full' if 'fullnorm' in exp_dict['img_dir'] else 'image' if 'norm' in exp_dict['img_dir'] else 'None'
        
        # Add pct stats
        # test_preds = 
        # pct_gr, pct_igr, pct_bp = feature_stats(test_preds)
        
        # Change str values for compact display
        if 'timeseries' in exp_dict['channels'][0]:
            exp_dict['channels'] = [exp_dict['channels'][0].replace('timeseries', 'TS')] 
        if len(exp_dict['channels']) == 2 and 'median_residual' in exp_dict['channels'][1]:
            exp_dict['channels'] = [exp_dict['channels'][0], 'MedRes'] 
            
            
        # Get name form folder, not dict
        exp_dict['WNet_name'] = expdir
        all_info.append(exp_dict)
    
    # Drop cols and sort 
    all_info = pd.DataFrame(all_info).drop(columns=['img_dir', 'seg_dir', 'img_size', 'num_sup', 'freeze_dec', 'batch_size', 'randomSharp','smooth_loss','blob_loss','reconstruct_mag', 'task_dir', 'sub_dir'], errors='ignore')
    all_info = all_info.sort_values(by='WNet_name')

    # Change col names for compact display
    all_info = all_info.rename(columns={"smooth_wght":"smth_wgt","blob_wght":"cohsn_wgt","ncut_wght":"nct_wgt","padding_mode":"pad_mode","learning_rate":"lr","num_epochs":"ne","kernel_size":"k_size"}) 
    
    return all_info 

def get_decDF(modeldir='../dec_runs/MURaM', tag=''):
    '''
    Helper function to create dataframe of all run models, thier parameters, and results 
    '''
    expdirs = [f for f in os.listdir(modeldir) if os.path.isdir(f'{modeldir}/{f}') and tag in f]
    
    # Create DF from exp dicts
    all_info = []
    for expdir in expdirs:
        
        # Skip if not finished training 
        if not os.path.exists(f'{modeldir}/{expdir}/test_preds_MURaM'):
            print(f'Skipping {expdir}; not finished training')
            continue
        if not os.path.exists(f'{modeldir}/{expdir}/exp_file.json'):
            print(f'Skipping {expdir}; no exp_file found')
            continue
        exp_dict = json.load(open(f'{modeldir}/{expdir}/exp_file.json','rb'))
        
        # Add val mse
        rmse = prediction_validation_results(output_dir=f'{modeldir}/{expdir}/test_preds_MURaM')
        exp_dict['RMSE'] = rmse
        
        # Add data norm
        exp_dict['norm'] = 'full' if 'fullnorm' in exp_dict['img_dir'] else 'image' if 'norm' in exp_dict['img_dir'] else 'None'
       
        # Add to dict
        all_info.append(exp_dict)
    
    # Drop cols and sort 
    all_info = pd.DataFrame(all_info).drop(columns=['img_dir', 'seg_dir', 'img_size','randomSharp','sub_dir'])
    all_info = all_info.sort_values(by='net_name')

    # Change col names for compact display
    all_info = all_info.rename(columns={"learning_rate":"lr","num_epochs":"ne"}) 
    
    return all_info 

def standardize_bin_preds(preds, name):
    '''
    Hacky way to ensure segmentationss have 0=igr, 1=gr, 1.5=bp 
    Can't think of a robust way to pick out which class has been assigned 1s or 0s, so just going to pass a list
    NOTE: this list only includes "resonable" segmentations, e.g. where there are *visually* clear gr, igr, and bp classes 
    '''
    gr_nums = {'WNetM_1E':1,'WNetX_bin1A':1, 'WNetX_bin1B':0,'WNetX_bin1C':0,'WNetXfn_bin1A':0,'WNetXfn_bin1C':0,'WNetXfn_bin1D':1}
    try:
        gr_num = gr_nums[name]
    except KeyError:
        raise ValueError(f'Need to add {name} to gr num dict')
    
    out = np.copy(preds)
    ig_num = 1 if gr_num == 0 else 0
    out[preds==ig_num] = 0.0
    out[preds==gr_num] = 1.0
        
    return out


def feature_stats(preds):
    '''
    Compute stats on feature distributions from given model results
    Preds should be [N_obs, N_pix, N_pix]
    '''
    
    preds = standardize_preds(preds) # Ensure segmentationss have 0=igr, 1=gr, 1.5=bp 
    a = np.array(preds).reshape(-1)
    pct_gr = len(a[a==0])/len(a)
    pct_igr = len(a[a==1])/len(a)
    pct_bp = len(a[a==1.5])/len(a)
    
    return pct_gr, pct_igr, pct_bp


def prediction_validation_results(output_dir, metric='MSE'):
    '''
    Compute average error on validation predictions 
    '''

    truefiles = [file for file in os.listdir(output_dir) if 'true' in file]
    predfiles = [file for file in os.listdir(output_dir) if 'pred' in file]
    tot_rmse = 0
    for i in range(len(truefiles)):
        true = np.load(f'{output_dir}/{truefiles[i]}').flatten()
        pred = np.load(f'{output_dir}/{predfiles[i]}').flatten()
        tot_rmse += np.sqrt(np.nanmean((true-pred)**2))
   
    out = tot_rmse/len(truefiles)
        
    return out


