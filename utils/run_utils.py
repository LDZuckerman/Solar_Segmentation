import numpy as np
import cv2
import scipy.ndimage as sndi
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import json
import loss_funcs

##############################################################
# Training and evaluation for WNet (can also be used w/ FrNet)
##############################################################

def train_wnet_batch(model, optimizer, input, k, img_size, smooth_wght=10, ncut_wght=10, blob_wght=1e-1, psi=0.5, device='cpu', train_enc_sup=False, labels=None, freeze_dec=False, target_pos=0, weights=None, debug=False, epoch=np.NaN):
    '''
    Train WNet on one batch of data
    '''
    softmax = nn.Softmax2d()
    smoothLoss = OpeningLoss2D()
    input = input.to(device) 

    enc = model(input, returns='enc') # predict seg of k="squeeze" classes 
    if eval(str(train_enc_sup)): # if running supervised (FrNet)
        bp_weight = 5 # dm_weight = 4
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1,1,bp_weight]))
        enc_loss = loss_fn(enc, labels)
    else:
        n_cut_loss = soft_n_cut_loss(input, softmax(enc),  img_size, debug) 
        if smooth_wght == 0: raise ValueError('WARNING: Not using smooth loss. Is this intentional?')
        # if (eval(str(smooth_loss)) and not eval(str(blob_loss))): # (8/6) REMOVE BOOLEEN LOSS TERM FLAGS - JUST WEIGHT AS 0
        #     enc_loss = smooth_wght*smoothLoss(softmax(enc)) + ncut_wght*n_cut_loss 
        # elif (eval(str(smooth_loss)) and eval(str(blob_loss))): # Why does this not seem to be doing anything now? 
        #     # print(f'Using blob_loss with weight {blob_wght}')
        #     # print(f'Loss terms')
        #     # print(f'  ncut: {ncut_wght}*{n_cut_loss}={ncut_wght*n_cut_loss}')
        #     # print(f'  smooth: {smooth_wght}*{smoothLoss(softmax(enc))}={smooth_wght*smoothLoss(softmax(enc))}')
        #     # print(f'  blob: {blob_wght}*{blobloss(enc)}={blob_wght*blobloss(enc)}')
        #     # if epoch == 2: a=b
        #     enc_loss = smooth_wght*smoothLoss(softmax(enc)) + ncut_wght*n_cut_loss + blob_wght*blobloss(enc)
        # else:
        #     print(f'smooth_loss: {smooth_loss} {type(smooth_loss)}, {eval(str(smooth_loss))}')
        #     raise ValueError('WARNING: Not using smooth loss. Is this intentional?')
        #     enc_loss = n_cut_loss 
        enc_loss = ncut_wght*n_cut_loss
        if blob_wght != 0:
            enc_loss +=  blob_wght*blobloss(enc) 
        if smooth_wght != 0:
            enc_loss += smooth_wght*smoothLoss(softmax(enc))
    if torch.isnan(enc_loss).any() == True: 
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(input[-1,target_pos,:,:]) # last img in batch, first channel
        axs[1].imshow(input[-1,1,:,:]) # last img in batch, second channel
        axs[2].imshow(np.argmax(enc[-1,:,:,:].detach().numpy(), axis=0)) # seg for last img in batch
        raise ValueError('enc loss has become NaN')
    enc_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if enc.shape[1] != k: 
        raise ValueError(f'Number of classes k is {k} but enc has shape {enc.shape}')
    example_seg = np.argmax(enc[-1,:,:,:].detach().numpy(), axis=0) # seg for last img in batch

    if eval(str(freeze_dec)):
        rec_loss = torch.tensor(np.NaN)
        example_rec = np.zeros_like(example_seg)*np.NaN
        example_rec2 = None
    else:
        dec = model(input, returns='dec') # predict image (all channels)
        rec_loss = multichannel_MSE_loss(input, dec, weights)  # from reproduction (MSELoss betwn input and rec imag) BUT with added channel weights
        if torch.isnan(rec_loss).any() == True: raise ValueError('rec loss has become NaN')
        rec_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        example_rec = dec[-1,target_pos,:,:].cpu().detach().numpy() # rec image for last img in batch 
        if dec.shape[1] == 2: # if only two channels, want to plot both
            example_rec2 = dec[-1,1,:,:].cpu().detach().numpy() # rec channel2 for last img in batch 
        else: example_rec2 = None 

    return model, enc_loss, rec_loss, example_seg, example_rec, example_rec2

def train_WNet(dataloader, wnet, optimizer, k, img_size, exp_outdir, WNet_id, smooth_wght, blob_wght, ncut_wght, epoch, device='cpu', train_enc_sup=False, freeze_dec=False, target_pos=0, weights=None, debug=False, save_examples=True):
    '''
    Train WNet for one epoch
    '''
    enc_losses = []
    rec_losses = []
    example_imgs = [] # yes.. could access through dataloader later instead of storing
    example_segs = []
    example_recs = []
    example_img2s = []
    example_rec2s = []

    # Train on each batch in train loader
    for (idx, batch) in enumerate(dataloader):
        print(f'\t   batch {idx}', end='\r', flush=True)
        X = batch[0] # batch is [images, labels]
        y = batch[1] # only used if train_enc_sup = True
        wnet, enc_loss, rec_loss, example_seg, example_rec, example_rec2, = train_wnet_batch(wnet, optimizer, X, k, img_size=img_size, smooth_wght=smooth_wght, ncut_wght=ncut_wght, blob_wght=blob_wght, device=device, train_enc_sup=train_enc_sup, labels=y, freeze_dec=freeze_dec, target_pos=target_pos, weights=weights, debug=debug, epoch=epoch)
        enc_losses.append(enc_loss.detach())
        rec_losses.append(rec_loss.detach())
        example_segs.append(example_seg)
        example_recs.append(example_rec)
        example_imgs.append(X[-1,target_pos,:,:]) # last img in batch, first channel
        if X.shape[1] == 2: # if two channels
            example_rec2s.append(example_rec2)
            example_img2s.append(X[-1,1,:,:])

    # Plot example imgs from each 10 random batches
    if save_examples:
        cols = 3 if len(example_img2s) == 0 else 5 # if I've stored second rec and image channels
        idxs = np.random.choice(np.linspace(0, len(example_segs)-1, len(example_segs)-1, dtype='int'), size=10)
        fig, axs = plt.subplots(10, cols, figsize=(cols*3, 10)) #plt.subplots(len(example_segs), cols, figsize=(cols*3, len(example_segs)))
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
                axs[i,4].imshow(example_rec2s[idx])
        for i in range(len(idxs)):
            for j in range(cols):
                axs[i,j].xaxis.set_tick_params(labelbottom=False); axs[i,j].yaxis.set_tick_params(labelleft=False); axs[i,j].set_xticks([]); axs[i,j].set_yticks([])
        plt.tight_layout()
        plt.savefig(f'{exp_outdir}/epoch{epoch}_examples'); plt.close()

    # Add losses to loss lists
    enc_losses.append(torch.mean(torch.FloatTensor(enc_losses)))
    rec_losses.append(torch.mean(torch.FloatTensor(rec_losses)))

    return enc_losses, rec_losses

def save_WNET_results(val_loader, save_dir, model, target_pos=0):
    '''
    Run each vtest obs through model, save results
    True and predicted vals are saved as 2d maps; e.g. compressed back to original seg format
    '''
    print(f'Loading model back in, saving results on test data in {save_dir}')
    if os.path.exists(save_dir) == False: os.mkdir(save_dir)
    i = 0
    for X, y in val_loader:

        X, y = X.to('cpu'), y.to('cpu')
        probs = model(X, returns='enc') # defualt is to return dec, but we want seg
        preds = np.argmax(probs.detach().numpy(), axis=1).astype(float) # turn probabilities [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
        y = np.argmax(y.detach().numpy(), axis=1).astype(float) # turn 1-hot-encoded layers [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
        if probs.shape[1] == 3: # if 3 classs (predictions have 4 layers)
            preds[preds == 2] = 1.5 # change from idx to class value 0 -> 0 (IG), 1 -> 1 (G), 2 -> 1.5 (BP)
            y[y == 2] = 1.5 # change from idx to class value 0 -> 0 (IG), 1 -> 1 (G), 2 -> 1.5 (BP)
        elif probs.shape[1] == 4: # if 4 classs (predictions have 4 layers)
            preds = preds/2 # change from idx to class value 0 -> 0 (IG), 1 -> 0.5 (DM), 2 -> 1 (G), 3 -> 1.5 (BP)
            y = y/2 # change from idx to class value 0 -> 0 (IG), 1 -> 0.5 (DM), 2 -> 1 (G), 3 -> 1.5 (BP)
        for j in range(np.shape(preds)[0]): # loop through batch 
            np.save(f'{save_dir}/x_{i}', np.array(X[j]))
            np.save(f'{save_dir}/true_{i}', np.array(y[j]))
            np.save(f'{save_dir}/pred_{i}', np.array(preds[j]))
            i += 1

##############################
# UNet training and evaluation
##############################

def train_UNET(loader, model, optimizer, loss_func, scaler, dm_weight=1, bp_weight=1, device='cpu'):
        #train_loader, model, optimizer, k=d['n_classes'], img_size=(d['img_size'], d['img_size']), exp_outdir=exp_outdir, UNet_id=UNet_id, epoch=epoch,  device=device, target_pos=target_pos, weights=d['weights'], save_examples=save_examples)
    '''
    Train UNet for one epoch
    '''
    
    loop = tqdm(loader) 

    # Train on each set in train loader
    for batch_idx, (data, targets) in enumerate(loop):
        # set to use device
        data = data.to(device)
        targets = targets.float().to(device)
        # forward
        predictions = model(data) 
        loss_fn = get_loss_func(loss_func)
        if isinstance(loss_fn, multiclass_MSE_loss):
            loss = loss_fn(predictions, targets, dm_weight, bp_weight) # compute loss
        elif isinstance(loss_fn, nn.CrossEntropyLoss):
            loss = loss_fn(predictions, targets) # compute loss
        else: 
            loss = loss_fn(predictions, targets) # compute loss
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def validate(val_loader, model, device='cpu'):
    '''
    Calculate validation accuracy after one training epoch
    '''
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval() # set model into eval mode
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)#.unsqueeze(1)
            preds = torch.sigmoid(model(x)) 
            if preds.shape[1] == 1: # if binary (predictions have 1 layer)
                preds = (preds > 0.5).float() # turn regression value (between zero and 1, e.g. "prob of being 1") into predicted 0s and 1s
            else: # if muliclasss (predictions have n_classes layers)
                preds = np.argmax(preds.detach().numpy(), axis=1) # turn probabilities [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
                y = np.argmax(y.detach().numpy(), axis=1) # turn 1-hot-encoded layers [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
            num_correct += len(np.where(preds == y)[0]) 
            num_pixels += len(preds.flatten()) 
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
    accuracy = num_correct/num_pixels*100

    return accuracy, dice_score

def save_UNET_results(val_loader, save_dir, model):
    '''
    Run each validation obs through model, save results
    True and predicted vals are saved as 2d maps; e.g. compressed back to original seg format
    '''
    print(f'Loading model back in, saving results on validation data in {save_dir}')
    if os.path.exists(save_dir) == False: os.mkdir(save_dir)
    i = 0
    for X, y in val_loader:
        X, y = X.to('cpu'), y.to('cpu')
        preds = torch.sigmoid(model(X))
        if preds.shape[1] == 2: # if binary (predictions have 2 layers) 
            preds = (preds > 0.5).float() # turn regression value (between zero and 1, e.g. "prob of being 1") into predicted 0s and 1s
            preds = preds.detach().numpy()[:,0,:,:] # binary inputs/truths are 2 layers where second is just dummy layer
            y = y.detach().numpy()[:,0,:,:] # binary inputs/truths are 2 layers where second is just dummy layer
        elif preds.shape[1] == 3: # if 3 classs (predictions have 4 layers)
            preds = np.argmax(preds.detach().numpy(), axis=1).astype(np.float) # turn probabilities [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
            y = np.argmax(y.detach().numpy(), axis=1).astype(np.float) # turn 1-hot-encoded layers [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
            preds[preds == 2] = 1.5 # change from idx to class value 0 -> 0 (IG), 1 -> 1 (G), 2 -> 1.5 (BP)
            y[y == 2] = 1.5 # change from idx to class value 0 -> 0 (IG), 1 -> 1 (G), 2 -> 1.5 (BP)
        elif preds.shape[1] == 4: # if 4 classs (predictions have 4 layers)
            preds = np.argmax(preds.detach().numpy(), axis=1) # turn probabilities [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
            y = np.argmax(y.detach().numpy(), axis=1) # turn 1-hot-encoded layers [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
            preds = preds/2 # change from idx to class value 0 -> 0 (IG), 1 -> 0.5 (DM), 2 -> 1 (G), 3 -> 1.5 (BP)
            y = y/2 # change from idx to class value 0 -> 0 (IG), 1 -> 0.5 (DM), 2 -> 1 (G), 3 -> 1.5 (BP)
        for j in range(np.shape(preds)[0]):
            np.save(f'{save_dir}/x_{i}', np.array(X[j]))
            np.save(f'{save_dir}/true_{i}', np.array(y[j]))
            np.save(f'{save_dir}/pred_{i}', np.array(preds[j]))
            i += 1

def compute_validation_results(output_dir, n_classes=2):
    '''
    Compute total percent correct, and percent correct on each class
    '''

    truefiles = [file for file in os.listdir(output_dir) if 'true' in file]
    predfiles = [file for file in os.listdir(output_dir) if 'pred' in file]

    pix_correct, ig_correct, dm_correct, gr_correct, bp_correct = 0, 0, 0, 0, 0
    tot_pix, tot_ig, tot_dm, tot_gr, tot_bp = 0, 0, 0, 0, 0

    for i in range(len(truefiles)):
        true = np.load(f'{output_dir}/{truefiles[i]}')
        preds = np.load(f'{output_dir}/{predfiles[i]}')
        pix_correct += len(np.where(preds.flatten() == true.flatten())[0]) #(preds == true).sum()
        tot_pix += len(preds.flatten()) # torch.numel(preds)
        if n_classes == 2: 
            ig_correct += len(np.where((preds.flatten() == true.flatten()) & (true.flatten() == 0))[0])
            tot_ig += len(np.where(true.flatten() == 0)[0])
            gr_correct += len(np.where((preds.flatten() == true.flatten()) & (true.flatten() == 1))[0])
            tot_gr += len(np.where(true.flatten() == 1)[0])
            dm_correct, tot_dm, bp_correct, tot_bp =  np.NaN, np.NaN, np.NaN, np.NaN
        if n_classes == 3: 
            ig_correct += len(np.where((preds.flatten() == true.flatten()) & (true.flatten() == 0))[0])
            tot_ig += len(np.where(true.flatten() == 0)[0])
            gr_correct += len(np.where((preds.flatten() == true.flatten()) & (true.flatten() == 1))[0])
            tot_gr += len(np.where(true.flatten() == 1)[0])
            bp_correct += len(np.where((preds.flatten() == true.flatten()) & (true.flatten() == 1.5))[0])
            tot_bp += len(np.where(true.flatten() == 1.5)[0])
            dm_correct, tot_dm =  np.NaN, np.NaN
        if n_classes == 4: 
            ig_correct += len(np.where((preds.flatten() == true.flatten()) & (true.flatten() == 0))[0])
            tot_ig += len(np.where(true.flatten() == 0)[0])
            dm_correct += len(np.where((preds.flatten() == true.flatten()) & (true.flatten() == 0.5))[0])
            tot_dm += len(np.where(true.flatten() == 0.5)[0])
            gr_correct += len(np.where((preds.flatten() == true.flatten()) & (true.flatten() == 1))[0])
            tot_gr += len(np.where(true.flatten() == 1)[0])
            bp_correct += len(np.where((preds.flatten() == true.flatten()) & (true.flatten() == 1.5))[0])
            tot_bp += len(np.where(true.flatten() == 1.5)[0])
    pct_correct = pix_correct/tot_pix*100
    pct_ig_correct = ig_correct/tot_ig*100
    pct_dm_correct = dm_correct/tot_dm*100
    pct_gr_correct = gr_correct/tot_gr*100
    pct_bp_correct = bp_correct/tot_bp*100

    return pct_correct, pct_ig_correct, pct_dm_correct, pct_gr_correct, pct_bp_correct

def eval_metrics(metric, true, preds):
    """
    Ways of evaluating extent to which true and preds are the same, if they should be
    E.g. for comparing labels to outputs of surpervised method, or to outputs of 2-value unsupervised method
    where those two values have been converted to 0 = IG, 1 = G.
    Only usefull for binary.
    """
    if metric == 'pct_correct':
        # Could be ok for our use; in gernal bad if huge class imbalance (could get high score by predicting all one class)
        return len(np.where(preds==true)[0])/len(preds)
    if metric == 'accuracy_score':
        # Avg of the area of overlap over area of union for each class (like Jaccard score but for two or more classes)
        return metrics.accuracy_score(true, preds)


def gradient_regularization(softmax):
    '''
    From https://github.com/AsWali/WNet/blob/master/utils (for use in soft_n_cut_loss)
    '''
    # Expects channels dim before batch dime?
    vertical_sobel=torch.nn.Parameter(torch.from_numpy(np.array([[[[1,  0,  -1], 
                                                                   [1,  0,  -1], 
                                                                   [1,  0,  -1]]]])).float(), requires_grad=False)
    horizontal_sobel=torch.nn.Parameter(torch.from_numpy(np.array([[[[1,   1,  1], 
                                                                     [0,   0,  0], 
                                                                     [-1 ,-1, -1]]]])).float(), requires_grad=False)
    vert=torch.cat([F.conv2d(softmax[:, i].unsqueeze(1), vertical_sobel) for i in range(softmax.shape[0])], 1)
    hori=torch.cat([F.conv2d(softmax[:, i].unsqueeze(1), horizontal_sobel) for i in range(softmax.shape[0])], 1)
    mag=torch.pow(torch.pow(vert, 2)+torch.pow(hori, 2), 0.5)
    mean=torch.mean(mag)

    return mean

def get_loss_func(loss_name):
    '''
    Load loss functions from loss_funcs.py
    '''

    loss_func = getattr(loss_funcs, loss_name)
    try:
        l=loss_func()
    except:
        l=loss_func

    return l

########################################
# Helper functions for evaluating models
########################################

def onehot_to_map(y):
    '''
    Helper function to turn 3D truth stack into 2D array of truth labels (also changes tensor to numpy)
    '''
    if y.shape[1] == 2: # if binary (y has 2 layers) 
        y = y.detach().numpy()[:,0,:,:] 
    elif y.shape[1] == 3: # if 3 classs (y has 3 layers)
        y = np.argmax(y.detach().numpy(), axis=1).astype(np.float) # turn 1-hot-encoded layers [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
        y[y == 2] = 1.5 # change from idx to class value 0 -> 0 (IG), 1 -> 1 (G), 2 -> 1.5 (BP)
    elif y.shape[1] == 4: # if 4 classs (y has 3 layers)
        y = np.argmax(y.detach().numpy(), axis=1) # turn 1-hot-encoded layers [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
        y = y/2 # change from idx to class value 0 -> 0 (IG), 1 -> 0.5 (DM), 2 -> 1 (G), 3 -> 1.5 (BP)

    return y


def probs_to_preds(probs):
    '''
    Helper function to turn 3D class probs into 2D arrays of predictions (also changes tensor to numpy)
    '''
    if probs.shape[1] == 1: # if binary (predictions have 1 layer)
        preds = (probs > 0.5).float() # turn regression value (between zero and 1, e.g. "prob of being 1") into predicted 0s and 1s
    else: # if muliclasss (predictions have n_classes layers)
        preds = np.argmax(probs.detach().numpy(), axis=1) # turn probabilities [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]

    return preds

def get_modelsDF(taskdir='../WNet_runs/MURaM', tag=''):
    '''
    Helper function to create dataframe of all run models and thier parameters
    '''
    expdirs = [f for f in os.listdir(taskdir) if os.path.isdir(f'{taskdir}/{f}') and tag in f]
    
    # Create DF from exp dicts
    all_info = []
    for expdir in expdirs:
        
        # Skip if not finished training 
        if not os.path.exists(f'{taskdir}/{expdir}/test_preds_MURaM'):
            print(f'Skipping {expdir}; not finished training')
            continue
        if not os.path.exists(f'{taskdir}/{expdir}/exp_file.json'):
            print(f'Skipping {expdir}; no exp_file found')
            continue
        exp_dict = json.load(open(f'{taskdir}/{expdir}/exp_file.json','rb'))
        
        # Add kernel size and task dir for models run before this was added as a changable parameter
        if 'kernel_size' not in exp_dict.keys():
            exp_dict['kernel_size'] = 3
        if 'task_dir' not in exp_dict.keys():
            exp_dict['task_dir'] = 'MURaM'
        
        # Change str values for compact display
        if 'timeseries' in exp_dict['channels'][0]:
            exp_dict['channels'] = [exp_dict['channels'][0].replace('timeseries', 'TS')] 
        if len(exp_dict['channels']) == 2 and 'median_residual' in exp_dict['channels'][1]:
            exp_dict['channels'] = [exp_dict['channels'][0], 'MedRes'] 
            
        # Get name form folder, not dict
        exp_dict['WNet_name'] = expdir
        all_info.append(exp_dict)
    
    # Drop cols and sort 
    all_info = pd.DataFrame(all_info).drop(columns=['img_dir', 'seg_dir', 'img_size', 'num_sup', 'freeze_dec', 'batch_size', 'randomSharp','smooth_loss','blob_loss'])
    all_info = all_info.sort_values(by='WNet_name')

    # Change col names for compact display
    all_info = all_info.rename(columns={"smooth_wght":"smth_wgt","blob_wght":"blb_wgt","ncut_wght":"nct_wgt", "padding_mode":"pad_mode","learning_rate":"lr","num_epochs":"ne","kernel_size":"k_size"}) 
    
    return all_info 
