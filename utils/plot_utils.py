import numpy as np
import cv2
import scipy.ndimage as sndi
import os
import matplotlib.pyplot as plt
import json
import torch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.colors import LogNorm

try:
    import loss_funcs, eval_utils
except ModuleNotFoundError:
    from utils import loss_funcs, eval_utils

    
###############
# WNet outputs
###############
    
def plot_preds_one_wnet(WNet_name):
    
    output_dir = f'../../model_runs_seg/MURaM/{WNet_name}/test_preds_MURaM'
    N = 3
    fig, axs = plt.subplots(N, 3, figsize=(8, N*2.5))
    axs[0,0].set_title('Image')
    axs[0,1].set_title('Algorithmic Labels')
    axs[0,2].set_title('WNet Labels')
    target_pos = 2 if 'WnetT' in WNet_name else 0
    for i in range(3): 
        idx = np.random.randint(0, len([file for file in os.listdir(output_dir) if file.startswith('x')]))
        im = np.load(f'{output_dir}/x_{idx}.npy')[target_pos]
        algseg = np.load(f'{output_dir}/true_{idx}.npy')
        preds = np.squeeze(np.load(f'{output_dir}/pred_{idx}.npy'))
        axs[i,0].imshow(im, cmap='gist_gray'); axs[i,0].set_ylabel(idx)
        axs[i,1].imshow(algseg, vmin=0, vmax=1.5, cmap='gist_gray')
        axs[i,2].imshow(preds, vmin=0, vmax=1.5, cmap='gist_gray')
        axs[i,0].xaxis.set_tick_params(labelbottom=False); axs[i,0].yaxis.set_tick_params(labelleft=False); axs[i,0].set_xticks([]); axs[i,0].set_yticks([])
        axs[i,1].xaxis.set_tick_params(labelbottom=False); axs[i,1].yaxis.set_tick_params(labelleft=False); axs[i,1].set_xticks([]); axs[i,1].set_yticks([])
        axs[i,2].xaxis.set_tick_params(labelbottom=False); axs[i,2].yaxis.set_tick_params(labelleft=False); axs[i,2].set_xticks([]); axs[i,2].set_yticks([])
    plt.suptitle(f'WNet {WNet_name}\n', fontsize=14)
    plt.tight_layout()
    
    return fig
    
def compare_wnet_preds_ugly(wnet_names):
    
    ts40idx = 108 # 58
    ts80idx = 20 # FIND A GOOD ONE
    nontsidx = 692 # 51, 131 # 161  # 131 # dont want to go through timeseries test sets to actually find the imageset that whose target image is idx 131 in the non-TS test set
    ts40idx_fn = 5 # 1, 5
    ts80idx_fn = 5 # 5, 3
    nontsidx_fn = 3822
    fig, axs = plt.subplots(len(wnet_names), 3, figsize=(9, 3*len(wnet_names)))
    for i in range(len(wnet_names)):
        output_dir = f'../../model_runs_seg/MURaM/{wnet_names[i]}/test_preds_MURaM'
        target_pos = 2 if wnet_names[i]=='WNet15m' else 0
        if 'tT' in wnet_names[i]:
            idx = ts40idx if 'T_1' in wnet_names[i] else ts80idx if 'T_2' in wnet_names[i] else ts40idx_fn if 'Tfn_1' in wnet_names[i] else ts80idx_fn if 'Tfn_2' in wnet_names[i] else 'error' 
        else:
            idx = nontsidx_fn if 'fn' in wnet_names[i] else nontsidx
        im = np.load(f'{output_dir}/x_{idx}.npy')[target_pos] # index to get image
        algseg = np.load(f'{output_dir}/true_{idx}.npy')
        preds = np.squeeze(np.load(f'{output_dir}/pred_{idx}.npy'))
        if int(preds[40, 60]) == 0: # try to mostly have black be zero
            preds_copy = np.copy(preds)
            preds[preds_copy == 0.0] = 1
            preds[preds_copy == 1.0] = 0 
        axs[i,2].set_title(f'{wnet_names[i]}')
        axs[i,0].imshow(im, cmap='gist_gray')
        axs[i,1].imshow(algseg, vmin=0, vmax=1.5, cmap='gist_gray')
        axs[i,2].imshow(preds, vmin=0, vmax=1.5, interpolation='none', cmap='tab10') #cmap='gist_gray')#cmap='tab10')
        axs[i,0].xaxis.set_tick_params(labelbottom=False); axs[i,0].yaxis.set_tick_params(labelleft=False); axs[i,0].set_xticks([]); axs[i,0].set_yticks([])
        axs[i,1].xaxis.set_tick_params(labelbottom=False); axs[i,1].yaxis.set_tick_params(labelleft=False); axs[i,1].set_xticks([]); axs[i,1].set_yticks([])
        axs[i,2].xaxis.set_tick_params(labelbottom=False); axs[i,2].yaxis.set_tick_params(labelleft=False); axs[i,2].set_xticks([]); axs[i,2].set_yticks([])
    
    return fig


def compare_wnet_preds(wnet_names, add_algseg=False):
    
    colors = LinearSegmentedColormap.from_list('', [plt.cm.tab10(0), plt.cm.tab10(9), plt.cm.tab10(6)]) #  (note that in tab10, order is dark blue, pink, light blue)
    descs = {'WNetX_bin1A':'binary segmentation\n',
             'WNetXfn_bin1A':'binary segmentation\n',
             'WNetX_bin1B':'binary segmentation\n with smooth loss',
             'WNetX_bin1C':'binary segmentation\n with smooth loss',
             'WNetXfn_bin1C':'binary segmentation\n with smooth loss',
             'WNetXfn_bin1D':'binary segmentation\n with smooth loss',
             'WNetX_1Ba':'WNet segmentation on\nphotometric image alone',
             'WNetX_2Ba':'WNet segmentation with\n"reflect" padding',
             'WNetX_2B':'WNet segmentation on\nphotometric image and\nresidual with median filtered image', # Huh? why is this image wrong now?
             'WNetX_3A':'WNet segmentation on\nphotometric image\nand image gradients',
             'WNetX_3B':'WNet segmentation on\nphotometric image and\nresidual with median filtered image', # THIS IS WRONG!!! gradients!!!
             'WNetT_1C':'WNet segmentation on\nimage time series (+/-40s)',
             'WNetTfn_1A':'WNet segmentation on\nimage time series (+/-40s)',
             'WNetTfn_1D':'WNet segmentation on\nimage time series (+/-40s)\nwith cohesion loss',
             'WNetT_1D':'WNet segmentation on\nimage time series (+/-80s)', # THIS IS WRONG!!! SHOULD BE 2C
             'WNetT_2C':'WNet segmentation on\nimage time series (+/-80s)',
             'WNetM_1E':'WNet segmentation on\nphotometric image\nand magentic flux image',
             'TrNetXfn_1Ac':'WNet segmentation with \npre-trained decoder',
             'WNetXfn_2D':'WNet segmentation on\nphotometric image',
             'WNetMfn_1E':'WNet segmentation on\nphotometric image\nand magentic flux image',
             'WNetX_1B': 'WNet segmentation with\n"zeros" padding',
             'WNetX_1Aa': 'WNet segmentation with\n"reflect" padding', 
             'WNetX_2Ba': 'WNet segmentation with\n"reflect" padding',
             'WNetX_2Aa': ''}
    if 'fn' in wnet_names[0]:
        ts40idx = 5
        ts80idx = 5
        nontsidx = 3822
    else:
        ts40idx = 108 #39 #108 to match with notsidx692 but for some reason WNetT_2C testing didnt save past 69??? 
        ts80idx = 38
        nontsidx = 692 # 58 # change this if I find a match..
    loadidx = ts40idx if 'T' in wnet_names[0] and '1' in wnet_names[0] else ts80idx if 'tT' and '2' in wnet_names[0] else nontsidx 
    im = np.load(f'../../model_runs_seg/MURaM/{wnet_names[0]}/test_preds_MURaM/x_{loadidx}.npy')[0] # im = np.load(f'../../model_runs_seg/MURaM/WNetT_1D/test_preds_MURaM/x_{ts40idx}.npy')[0]
    n_ax = len(wnet_names)+2 if add_algseg else len(wnet_names)+1
    fig, axs = plt.subplots(1, n_ax, figsize=(4.5*(n_ax-1),4.5))
    axs[0].imshow(im, cmap='gist_gray')
    axs[0].set_title('photometric image\n', fontsize=14)
    if add_algseg:
        algseg = np.load(f'../../model_runs_seg/MURaM/WNetT_1D/test_preds_MURaM/true_{ts40idx}.npy')
        axs[-1].imshow(algseg, vmin=0, vmax=1.5, interpolation='none', cmap='gist_gray')#cmap=colors)
        axs[-1].set_title('segmentation with\nhueristc algorithm', fontsize=14)
    for i in range(len(wnet_names)):
        idx = ts40idx if 'T' in wnet_names[i] and '1' in wnet_names[i] else ts80idx if ('tT' in wnet_names[i] and '2' in wnet_names[i]) else nontsidx 
        preds = np.squeeze(np.load(f'../../model_runs_seg/MURaM/{wnet_names[i]}/test_preds_MURaM/pred_{idx}.npy'))
        out = make_pretty(wnet_names[i], preds)
        plot_idx = i+1 #i+2 if add_algseg else i+1
        try:
            desc = descs[wnet_names[i]]
        except KeyError:
            raise ValueError(f'Need to add description for {wnet_names[i]} to model descriptions in plot_utils.compare_wnet_preds()!')
        axs[plot_idx].set_title(desc, fontsize=14)
        axs[plot_idx].imshow(out, vmin=0, vmax=1.5, interpolation='none', cmap='gist_gray')#cmap=colors) 
    for ax in axs.flat:
        ax.xaxis.set_tick_params(labelbottom=False); ax.yaxis.set_tick_params(labelleft=False); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    return fig


def make_pretty(wnet_name, preds):
    '''
    This is a really hacky way to *try* to set dark blue for igr, light blue gr, pink bp [for some reason class nums are same for all T and all M models]
    '''
    out = np.copy(preds)
    if wnet_name == 'WNetT_1C':
        out[preds == 1.5] = 0.0
        out[preds == 0.0] = 1.0
        out[preds == 1.0] = 1.5
    elif wnet_name == 'WNetT_1D':
        out[preds == 1.5] = 1.0
        out[preds == 1.0] = 1.5
    # if 'T' in wnet_name:
    #     out[preds == 1.5] = 0.0
    #     out[preds == 0.0] = 1.0
    #     out[preds == 1.0] = 1.5
    elif 'M' in wnet_name:
        out[preds == 1.0] = 0.0
        out[preds == 1.5] = 1.0
        out[preds == 0.0] = 1.5
    elif wnet_name in ['WNetX_2B','WNetTfn_1D']:
        out[preds == 1.0] = 0.0
        out[preds == 0.0] = 1.0
    elif 'bin' in wnet_name:
        out = eval_utils.standardize_bin_preds(preds, wnet_name)
    elif wnet_name == 'WNetX_3B':
        out[preds == 1.0] = 1.5
        out[preds == 1.5] = 1.0
    elif 'X' in wnet_name:
        out[preds == 1.5] = 0.0
        out[preds == 0.0] = 1.0
        out[preds == 1.0] = 1.5
    else:
        print(f'Need to add re-map of {wnet_name} preds to plot_utils.make_pretty()!')
        
    return out


def plot_loss_one_wnet(wnet_name):
    
    rec_losses = np.load(f'../../model_runs_seg/MURaM/{wnet_name}/rec_losses.npy')
    ncut_losses = np.load(f'../../model_runs_seg/MURaM/{wnet_name}/n_cut_losses.npy')
    fig, ax = plt.subplots(1,1,figsize=(6,3))
    ax.plot(rec_losses, color='b', label='rec losses')
    ax1 = ax.twinx()
    ax1.plot(ncut_losses, color='g', label='n cut losses')
    ax.legend()
    ax1.legend(loc='upper center')
    plt.suptitle(wnet_name)
    
    return fig


def compare_wnet_loss(wnet_names, loss_term='both'):
    
    fig, ax = plt.subplots(1,1,figsize=(8,3))
    for i in range(len(wnet_names)):
        rec_losses = np.load(f'../../model_runs_seg/MURaM/{wnet_names[i]}/rec_losses.npy')
        ncut_losses = np.load(f'../../model_runs_seg/MURaM/{wnet_names[i]}/n_cut_losses.npy')
        if loss_term == 'both':
            ax.plot((rec_losses + ncut_losses)/np.max(rec_losses + ncut_losses), label=wnet_names[i])
        elif loss_term == 'rec':
            ax.plot(rec_losses/np.max(rec_losses), label=wnet_names[i])
        elif loss_term == 'ncut':
            ax.plot(ncut_losses/np.max(ncut_losses), label=wnet_names[i])
    plt.legend(fontsize=6)
    title = 'Sum of normalized loss terms' if loss_term == 'both' else f'Normalized {typ} loss'
    plt.title(title)
    

##################################
# Evaluation during model training
##################################

def plot_epoch_examples(data, targets, predictions, save_dir, epoch):
    '''
    Plot examples from last training batch of given epoch
    Include all layers of input, true, and pred
    Should be useful for any model
    '''
    if targets.ndim == 3: # if this is a seg thats already been collapsed
        targets = torch.unsqueeze(targets, 1)
        predictions = torch.unsqueeze(predictions, 1)
    X = data.cpu().detach()
    y = targets.cpu().detach()
    pred = predictions.cpu().detach().numpy() 
    idxs = np.random.choice(np.linspace(0, X.shape[0]-1, X.shape[0]-1, dtype='int'), size=5)
    in_layers = X.shape[1]
    out_layers = y.shape[1]
    n_cols = in_layers + 2*out_layers
    fig, axs = plt.subplots(5, n_cols, figsize=(n_cols*3, 10)) #plt.subplots(len(example_segs), cols, figsize=(cols*3, len(example_segs)))
    for i in range(len(idxs)): 
        for j in range(in_layers):
            im = axs[i,j].imshow(X[idxs[i],j,:,:]); plt.colorbar(im, ax=axs[i,j]) # vmin=0, vmax=1 # ith img in batch, jth channel
            axs[0,j].set_title(f'x [layer {j}]')
        for j in range(out_layers):
            im = axs[i,in_layers+j].imshow(y[idxs[i],j,:,:]); plt.colorbar(im, ax=axs[i,in_layers+j]) 
            axs[0,in_layers+j].set_title(f'true [layer {j}]')
        for j in range(out_layers):
            im = axs[i,in_layers+out_layers+j].imshow(pred[idxs[i],j,:,:]); plt.colorbar(im, ax=axs[i,in_layers+out_layers+j]) 
            axs[0,in_layers+out_layers+j].set_title(f'pred [layer {j}]')
        axs[i,0].set_ylabel(f'{idxs[i]}')
    plt.title(f'Examples from epoch {epoch} last train batch')
    plt.savefig(f'{save_dir}/epoch{epoch}_examples')
    

def plot_epoch_exmples_wnet(data, enc, rec, save_dir, epoch):
    '''
    Plot examples from last training batch of given epoch
    Include all layers of input, but collapse segmentations (dec) from onehot layers into map
    '''
    X = data.cpu().detach()
    enc = enc.cpu().detach().numpy()
    rec = rec.cpu().detach().numpy() 
    idxs = np.random.choice(np.linspace(0, X.shape[0]-1, X.shape[0]-1, dtype='int'), size=4) 
    in_layers = X.shape[1]
    n_cols = 2*in_layers + 1
    fig, axs = plt.subplots(5, n_cols, figsize=(n_cols*3, 10)) #plt.subplots(len(example_segs), cols, figsize=(cols*3, len(example_segs)))
    for i in range(len(idxs)): 
        for j in range(in_layers):
            im = axs[i,j].imshow(X[idxs[i],j,:,:]); plt.colorbar(im, ax=axs[i,j]) # vmin=0, vmax=1 # ith img in batch, jth channel
            axs[0,j].set_title(f'x [layer {j}]')
        y = np.argmax(enc[idxs[i],:,:,:], axis=0) # collapse to map by taking argmax along class dimension
        im1 = axs[i,in_layers].imshow(y); plt.colorbar(im1, ax=axs[i,in_layers]) 
        axs[0,in_layers].set_title(f'seg (argmax enc)')
        for j in range(in_layers):
            im = axs[i,in_layers+1+j].imshow(rec[idxs[i],j,:,:]); plt.colorbar(im, ax=axs[i,in_layers+1+j])
            axs[0,in_layers+1+j].set_title(f'rec [layer {j}]')
        axs[i,0].set_ylabel(f'{idxs[i]}')
    plt.suptitle(f'Examples from epoch {epoch} last train batch')
    plt.savefig(f'{save_dir}/epoch{epoch}_examples')

def plot_epoch_examples_wnet_allbatch():
    '''
    Outdated version of above, in which examples are plotted from each batch in epoch, not just last batch
    In order to plot examples from each batch (instead of final batch only), expects lists of example imgs, segs, and recs saved from each batch
        These are annoying to save 
    '''
    N = 5
    cols = 3 if (len(example_img2s) == 0 or 'bin' in exp_outdir) else 5 # if I've stored second rec and image channels
    idxs = np.random.choice(np.linspace(0, len(example_segs)-1, len(example_segs)-1, dtype='int'), size=N)
    fig, axs = plt.subplots(N, cols, figsize=(cols*3, N)) #plt.subplots(len(example_segs), cols, figsize=(cols*3, len(example_segs)))
    axs[0, 0].set_title('last img, targ ch')                     
    if cols == 3: # e.g. for TS or X alone, where only plotting one input channel
        axs[0, 1].set_title('seg (argmax enc)')
        axs[0, 2].set_title('rec, targ ch')
        for i in range(len(idxs)):
            idx =  idxs[i]   
            axs[i,0].set_ylabel(f'Batch {idx}')
            axs[i,0].imshow(example_imgs[idx], vmin=0, vmax=1) #(X[-1,i,:,:])
            axs[i,1].imshow(example_segs[idx])
            axs[i,2].imshow(example_recs[idx])
    if cols == 5: 
        axs[0, 1].set_title('last img, 2nd ch')
        axs[0, 2].set_title('seg (argmax enc)')
        axs[0, 3].set_title('rec, targ ch')
        axs[0, 4].set_title('rec, 2nd ch')
        for i in range(len(idxs)):
            idx =  idxs[i]  
            axs[i,0].set_ylabel(f'Batch {idx}')
            axs[i,0].imshow(example_imgs[idx], vmin=0, vmax=1) #(X[-1,i,:,:])
            axs[i,1].imshow(example_img2s[idx], vmin=0, vmax=1)
            axs[i,2].imshow(example_segs[idx])
            axs[i,3].imshow(example_recs[idx])
            try:
                axs[i,4].imshow(example_rec2s[idx])
            except TypeError:
                raise ValueError(f'example_rec2s[idx] is {example_rec2s[idx]}, which is {type(example_rec2s[idx])}')          
    for i in range(len(idxs)):
        for j in range(cols):
            axs[i,j].xaxis.set_tick_params(labelbottom=False); axs[i,j].yaxis.set_tick_params(labelleft=False); axs[i,j].set_xticks([]); axs[i,j].set_yticks([])
    plt.tight_layout()
    plt.savefig(f'{exp_outdir}/epoch{epoch}_examples'); plt.close()