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
try:
    import loss_funcs, plot_utils
except ModuleNotFoundError:
    from utils import loss_funcs, plot_utils

##############################################################
# Training and evaluation for WNet (can also be used w/ FrNet)
##############################################################

def train_wnet_batch(model, optimizer, input, k, img_size, smooth_wght=10, ncut_wght=10, cohesion_wght=1e-1, cont_wght=10, psi=0.5, device='cpu', labels=None, freeze_dec=False, target_pos=0, weights=None, debug=False, reconstruct_mag=False, epoch=np.NaN, temp_dir='None'):
    '''
    Train WNet on one batch of data
    '''
    
    #softmax = nn.Softmax2d()
    soft_n_cut_loss = get_loss_func('soft_n_cut_loss')
    smoothLoss = get_loss_func('OpeningLoss2D')
    cohesionLoss = get_loss_func('cohesion_loss')
    continuityLoss = get_loss_func('continuity_loss')
    multichannel_MSE_loss = get_loss_func('multichannel_MSE_loss')
    input = input.to(device)

    # Compute predicted segmentation and segmentation loss
    if reconstruct_mag:
        model_input = input[:-1,:,:] # remove mag flux that is tacked on 
    else:
        model_input = input
    enc = model(model_input, compute_dec=False)  # CLASS PROBS (NOT MASKS), already softmax # predict seg of k classes 
    n_cut_loss = soft_n_cut_loss(input, enc, img_size) # if reconstruct_mag, include mag channel in assesment of intra-cluster similarity
    enc_loss = ncut_wght*n_cut_loss
    if cohesion_wght != 0:
        enc_loss += cohesion_wght*cohesionLoss(enc, temp_dir=temp_dir) 
    if smooth_wght != 0:
        enc_loss += smooth_wght*smoothLoss(enc)
    if cont_wght != 0:
        enc_loss += cont_wght*continuityLoss(enc)
    if torch.isnan(enc_loss).any() == True: 
        raise ValueError(f'enc loss has become NaN\n n_cut_loss {n_cut_loss}\chsn_loss {cohesionLoss(enc, temp_dir=temp_dir)}, smoothLoss {smoothLoss(enc)}')
    enc_loss.backward() 
    optimizer.step()
    optimizer.zero_grad()
    #example_seg = np.argmax(enc[-1,:,:,:].cpu().detach().numpy(), axis=0) # seg for last img in batch

    # Compute reconstruction and reconstruction loss
    rec = model(model_input, compute_dec=True) # predict image (all channels)
    rec_loss = multichannel_MSE_loss(input, rec, weights) # if reconstruct_mag, include mag channel in assesment of reconstruction accuracy (rec will have n_channels + 1 layers)   # from reproduction (MSELoss betwn input and rec imag) BUT with added channel weights
    if torch.isnan(rec_loss).any() == True: raise ValueError('rec loss has become NaN')
    rec_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # example_rec = rec[-1,target_pos,:,:].cpu().detach().numpy() # rec image for last img in batch 
    # if rec.shape[1] == 2 and not reconstruct_mag: # if only two channels, want to plot both
    #     example_rec2 = rec[-1,1,:,:].cpu().detach().numpy() # rec channel2 for last img in batch 
    # else: example_rec2 = None 

    return model, enc_loss, rec_loss, enc, rec #, example_seg, example_rec, example_rec2

def train_WNet(dataloader, wnet, optimizer, k, img_size, exp_outdir, WNet_id, smooth_wght, cohesion_wght, ncut_wght, cont_wght, epoch, device='cpu', freeze_dec=False, target_pos=0, weights=None, debug=False, reconstruct_mag=False, save_examples=True, temp_dir='None'):
    '''
    Train WNet for one epoch
    '''
    enc_losses = []
    rec_losses = []
    # example_imgs = [] # yes.. could access through dataloader later instead of storing
    # example_segs = []
    # example_recs = []
    # example_img2s = []
    # example_rec2s = []

    # Train on each batch in train loader
    for (idx, batch) in enumerate(dataloader):
        print(f'\t   batch {idx}', end='\r', flush=True)# print(f'\t   batch {idx}', end='\r', flush=True)
        X = batch[0] # batch is [images, labels]
        #wnet, enc_loss, rec_loss, example_seg, example_rec, example_rec2 = train_wnet_batch(wnet, optimizer, X, k, img_size=img_size, smooth_wght=smooth_wght, ncut_wght=ncut_wght, blob_wght=blob_wght, device=device, labels=y, freeze_dec=freeze_dec, target_pos=target_pos, weights=weights, debug=debug, reconstruct_mag=eval(str(reconstruct_mag)), epoch=epoch, temp_dir=temp_dir)
        wnet, enc_loss, rec_loss, enc, dec = train_wnet_batch(wnet, optimizer, X, k, img_size=img_size, smooth_wght=smooth_wght, ncut_wght=ncut_wght, cohesion_wght=cohesion_wght, device=device, freeze_dec=freeze_dec, target_pos=target_pos, weights=weights, debug=debug, reconstruct_mag=eval(str(reconstruct_mag)), epoch=epoch, temp_dir=temp_dir)
        enc_losses.append(enc_loss.cpu().detach())
        rec_losses.append(rec_loss.cpu().detach())
        # example_segs.append(example_seg)
        # example_recs.append(example_rec)
        # example_imgs.append(X[-1,target_pos,:,:]) # last img in batch, first channel
        # if X.shape[1] == 2: # if two channels
        #     example_rec2s.append(example_rec2)
        #     example_img2s.append(X[-1,1,:,:])

    # Plot example imgs from last batch [NOTE: possible that last batch has less than 16 images]
    if save_examples:
        plot_utils.plot_epoch_exmples_wnet(data=X, enc=enc, rec=dec, save_dir=exp_outdir, epoch=epoch) # X, enc, and dec will be from last batch in above loop
        # N = 5
        # cols = 3 if (len(example_img2s) == 0 or 'bin' in exp_outdir) else 5 # if I've stored second rec and image channels
        # idxs = np.random.choice(np.linspace(0, len(example_segs)-1, len(example_segs)-1, dtype='int'), size=N)
        # fig, axs = plt.subplots(N, cols, figsize=(cols*3, N)) #plt.subplots(len(example_segs), cols, figsize=(cols*3, len(example_segs)))
        # axs[0, 0].set_title('last img, targ ch')                     
        # if cols == 3: # e.g. for TS or X alone, where only plotting one input channel
        #     axs[0, 1].set_title('seg (argmax enc)')
        #     axs[0, 2].set_title('rec, targ ch')
        #     for i in range(len(idxs)):
        #         idx =  idxs[i]   
        #         axs[i,0].set_ylabel(f'Batch {idx}')
        #         axs[i,0].imshow(example_imgs[idx], vmin=0, vmax=1) #(X[-1,i,:,:])
        #         axs[i,1].imshow(example_segs[idx])
        #         axs[i,2].imshow(example_recs[idx])
        # if cols == 5: 
        #     axs[0, 1].set_title('last img, 2nd ch')
        #     axs[0, 2].set_title('seg (argmax enc)')
        #     axs[0, 3].set_title('rec, targ ch')
        #     axs[0, 4].set_title('rec, 2nd ch')
        #     for i in range(len(idxs)):
        #         idx =  idxs[i]  
        #         axs[i,0].set_ylabel(f'Batch {idx}')
        #         axs[i,0].imshow(example_imgs[idx], vmin=0, vmax=1) #(X[-1,i,:,:])
        #         axs[i,1].imshow(example_img2s[idx], vmin=0, vmax=1)
        #         axs[i,2].imshow(example_segs[idx])
        #         axs[i,3].imshow(example_recs[idx])
        #         try:
        #             axs[i,4].imshow(example_rec2s[idx])
        #         except TypeError:
        #             raise ValueError(f'example_rec2s[idx] is {example_rec2s[idx]}, which is {type(example_rec2s[idx])}')          
        # for i in range(len(idxs)):
        #     for j in range(cols):
        #         axs[i,j].xaxis.set_tick_params(labelbottom=False); axs[i,j].yaxis.set_tick_params(labelleft=False); axs[i,j].set_xticks([]); axs[i,j].set_yticks([])
        # plt.tight_layout()
        # plt.savefig(f'{exp_outdir}/epoch{epoch}_examples'); plt.close()

    # Add losses to loss lists
    enc_losses.append(torch.mean(torch.FloatTensor(enc_losses)))
    rec_losses.append(torch.mean(torch.FloatTensor(rec_losses)))

    return enc_losses, rec_losses

def save_WNET_results(val_loader, save_dir, model, target_pos=0, device='cpu'):
    '''
    Run each vtest obs through model, save results
    True and predicted vals are saved as 2d maps; e.g. compressed back to original seg format
    '''
    print(f'Loading model back in, saving results on test data in {save_dir}')
    if os.path.exists(save_dir) == False: os.mkdir(save_dir)
    i = 0
    for X, y in val_loader:

        X, y = X.to(device), y.to(device)
        probs = model(X, compute_dec=False) # defualt is to return dec, but we want seg
        preds = np.argmax(probs.cpu().detach().numpy(), axis=1).astype(float) # turn probabilities [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
        y = np.argmax(y.cpu().detach().numpy(), axis=1).astype(float) # turn 1-hot-encoded layers [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
        if probs.shape[1] == 3: # if 3 classs (predictions have 4 layers)
            preds[preds == 2] = 1.5 # change from idx to class value 0 -> 0 (IG), 1 -> 1 (G), 2 -> 1.5 (BP)
            y[y == 2] = 1.5 # change from idx to class value 0 -> 0 (IG), 1 -> 1 (G), 2 -> 1.5 (BP)
        elif probs.shape[1] == 4: # if 4 classs (predictions have 4 layers)
            preds = preds/2 # change from idx to class value 0 -> 0 (IG), 1 -> 0.5 (DM), 2 -> 1 (G), 3 -> 1.5 (BP)
            y = y/2 # change from idx to class value 0 -> 0 (IG), 1 -> 0.5 (DM), 2 -> 1 (G), 3 -> 1.5 (BP)
        for j in range(np.shape(preds)[0]): # loop through batch 
            np.save(f'{save_dir}/x_{i}',  X[j].cpu().detach().numpy())
            np.save(f'{save_dir}/true_{i}', np.array(y[j]))
            np.save(f'{save_dir}/pred_{i}', np.array(preds[j]))
            i += 1

####################################
# Supervised training and evaluation
####################################

def train_net(loader, model, loss_func, loss_name, optimizer, chan_weights='None', device='cpu', save_examples=False, save_dir=None, epoch=None):
    '''
    Train superised for one epoch
    '''
    
    #loop = tqdm(loader) 

    # Train on each batch in train loader
    k = 0
    for data, targets in loader: #batch_idx, (data, targets) in enumerate(loop):
        print(f'  Batch {k}', end='\r', flush=True); k += 1
        # set to use device
        data = data.to(device)
        targets = targets.float().to(device)
        # forward
        predictions = model(data)
        if torch.isnan(predictions).any() or not torch.isfinite(predictions).all():
            raise ValueError('preds become NaN or inf')
        if loss_name == 'multichannel_MSE_loss':
            loss = loss_func(predictions, targets, weights=chan_weights)
        elif isinstance(loss_func, loss_funcs.multiclass_MSE_loss):
            loss = loss_func(predictions, targets, bp_weight=bp_weight) # compute loss # dm_weight
        elif 'enc' in save_dir:
            #predictions = torch.argmax(predictions, dim=1) # take argmax along class axis to trun class probs into seg map
            #targets = torch.argmax(targets, dim=1) # take argmax along class axis to trun class masks into seg map
            loss = loss_func(predictions, targets) # should be ok to compute MSE even when targets are masks and preds are probs, right?
        else: 
            loss = loss_func(predictions, targets) # compute loss
        loss.backward() # why was i only doing this once per batch???
        optimizer.step() # why did i not have this???
        optimizer.zero_grad() # why was i only doing this once per batch???
 
        
    # Plot examples from last batch 
    if save_examples: 
        plot_utils.plot_epoch_examples(data, torch.tensor(targets), torch.tensor(predictions), save_dir, epoch) # data, targets will be those loaded from last batch 
        # X = data.cpu().detach()
        # y = targets.cpu().detach()
        # pred = predictions.cpu().detach().numpy() 
        # idxs = np.random.choice(np.linspace(0, X.shape[0]-1, X.shape[0]-1, dtype='int'), size=5)
        # in_layers = X.shape[1]
        # out_layers = y.shape[1]
        # n_cols = in_layers + 2*out_layers
        # fig, axs = plt.subplots(5, n_cols, figsize=(n_cols*3, 10)) #plt.subplots(len(example_segs), cols, figsize=(cols*3, len(example_segs)))
        # for i in range(len(idxs)): 
        #     for j in range(in_layers):
        #         im = axs[i,j].imshow(X[idxs[i],j,:,:]); plt.colorbar(im, ax=axs[i,j]) # vmin=0, vmax=1 # ith img in batch, jth channel
        #         axs[0,j].set_title(f'x [layer {j}]')
        #     for j in range(out_layers):
        #         im = axs[i,in_layers+j].imshow(y[idxs[i],j,:,:]); plt.colorbar(im, ax=axs[i,in_layers+j]) # first y in batch, already class-collapsed
        #         axs[0,in_layers+j].set_title(f'true [layer {j}]')
        #     for j in range(out_layers):
        #         im = axs[i,in_layers+out_layers+j].imshow(pred[idxs[i],j,:,:]); plt.colorbar(im, ax=axs[i,in_layers+out_layers+j]) # first y in batch, already class-collapsed
        #         axs[0,in_layers+out_layers+j].set_title(f'pred [layer {j}]')
        #     # axs[i,0].set_ylabel(f'{idxs[i]}')
        #     # axs[i,0].imshow(X[idxs[i],0,:,:]) # this will be as set during training of last batch
        #     # axs[i,1].imshow(y[idxs[i],0,:,:])
        #     # axs[i,2].imshow(pred[idxs[i],0,:,:])  
        # plt.title(f'Examples from epoch {epoch} last train batch')
        # plt.savefig(f'{save_dir}/epoch{epoch}_examples')
        #if epoch==2 and k>90: print(f'  -> saved preds', flush=True)
        # backward
        # optimizer.zero_grad()
        # # scaler.scale(loss).backward()
        # # scaler.step(optimizer)
        # # scaler.update()
        # loss.backward()
        # # # update tqdm loop
        # # loop.set_postfix(loss=loss.item())
        # #if epoch==2 and k>100: print(f'  -> updated optimizer and loss', flush=True)
        
    return loss


'''
Helper functions
'''

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
                preds = np.argmax(preds.cpu().detach().numpy(), axis=1) # turn probabilities [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
                y = np.argmax(y.cpu().detach().numpy(), axis=1) # turn 1-hot-encoded layers [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
            num_correct += len(np.where(preds == y)[0]) 
            num_pixels += len(preds.flatten()) 
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
    accuracy = num_correct/num_pixels*100

    return accuracy, dice_score

def save_UNET_results(val_loader, save_dir, model, device='cpu'):
    '''
    Run each validation obs through model, save results
    True and predicted vals are saved as 2d maps; e.g. compressed back to original seg format
    '''
    print(f'Loading model back in, saving results on validation data in {save_dir}')
    if os.path.exists(save_dir) == False: os.mkdir(save_dir)
    i = 0
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        preds = torch.sigmoid(model(X))
        if preds.shape[1] == 2: # if binary (predictions have 2 layers) 
            preds = (preds > 0.5).float() # turn regression value (between zero and 1, e.g. "prob of being 1") into predicted 0s and 1s
            preds = preds.cpu().detach().numpy()[:,0,:,:] # binary inputs/truths are 2 layers where second is just dummy layer
            y = y.cpu().detach().numpy()[:,0,:,:] # binary inputs/truths are 2 layers where second is just dummy layer
        elif preds.shape[1] == 3: # if 3 classs (predictions have 4 layers)
            preds = np.argmax(preds.cpu().detach().numpy(), axis=1).astype(np.float) # turn probabilities [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
            y = np.argmax(y.cpu().detach().numpy(), axis=1).astype(np.float) # turn 1-hot-encoded layers [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
            preds[preds == 2] = 1.5 # change from idx to class value 0 -> 0 (IG), 1 -> 1 (G), 2 -> 1.5 (BP)
            y[y == 2] = 1.5 # change from idx to class value 0 -> 0 (IG), 1 -> 1 (G), 2 -> 1.5 (BP)
        elif preds.shape[1] == 4: # if 4 classs (predictions have 4 layers)
            preds = np.argmax(preds.cpu().detach().numpy(), axis=1) # turn probabilities [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
            y = np.argmax(y.cpu().detach().numpy(), axis=1) # turn 1-hot-encoded layers [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
            preds = preds/2 # change from idx to class value 0 -> 0 (IG), 1 -> 0.5 (DM), 2 -> 1 (G), 3 -> 1.5 (BP)
            y = y/2 # change from idx to class value 0 -> 0 (IG), 1 -> 0.5 (DM), 2 -> 1 (G), 3 -> 1.5 (BP)
        for j in range(np.shape(preds)[0]):
            np.save(f'{save_dir}/x_{i}',  X[j].cpu().detach().numpy())
            np.save(f'{save_dir}/true_{i}', np.array(y[j]))
            np.save(f'{save_dir}/pred_{i}', np.array(preds[j]))
            i += 1
            
def save_model_results(val_loader, save_dir, model, device='cpu'):
    '''
    Run each validation obs through model, save results
    '''
    print(f'Loading model back in, saving results on validation data in {save_dir}')
    if os.path.exists(save_dir) == False: os.mkdir(save_dir)
    i = 0
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        if torch.is_tensor(y):
            y = y.cpu().detach().numpy()
        preds = model(X).cpu().detach().numpy()
        for j in range(np.shape(preds)[0]):
            np.save(f'{save_dir}/x_{i}', X[j].cpu().detach().numpy())
            np.save(f'{save_dir}/true_{i}', np.array(y[j]))
            np.save(f'{save_dir}/pred_{i}', np.array(preds[j]))
            i += 1            
            
def segmentation_validation_results(output_dir, n_classes=2):
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

def get_ch_mul(depth, img_size):
    '''
    Calculate appropriate ch_mul for UDec to ensure that we dont run out of pixels
    - Must keep ch_mul * 2**(depth+1) < n_pix**2
    '''
    
    ch_mul = 64 # good starting point
    while ch_mul >= img_size**2 / 2**(depth+1):  # adding 1 to depth since I think max_pool with kernel_size = (2,2) should also divide in half? 
        print('ch_mul', ch_mul)
        ch_mul = int(ch_mul/2)
        if ch_mul < 8:
            #raise ValueError(f'With images of size {img_size}**2, would need first down conv to produce only 4 layers in order to have depth {depth}.'+
                             #'This seems too low, so should probabaly use a smaller depth.')
            print('ERROR');a=b
    
    return ch_mul