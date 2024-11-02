import torch.nn as nn
import numpy as np
import cv2
import scipy.ndimage as sndi
import torch
import torch.nn.functional as F
import os

'''
Loss functions for both supervised and WNet models
'''
# NOTE: what about also trying something that charecterizes the degree to which the pixels in one class are brighter than the pixels in the second brightest class?


def multichannel_MSE_loss(x, x_prime, weights):
    '''
    MSE loss but with ability to weight loss from each channel differently
    For WNet reconstruction loss
    '''

    loss = 0
    for channel in range(x.shape[1]):
        mse = nn.MSELoss()(x[:,channel,:,:], x_prime[:,channel,:,:])
        loss += weights[channel]*mse
    loss = loss/(x.shape[1])
    
    if torch.isnan(loss).any() or not torch.isfinite(loss).all():
        raise ValueError('loss has become NaN or inf')

    return loss

    
def cohesion_loss(outputs, temp_dir):
    '''
    Metric describing how much the classes group pixels into circular blobs vs long shapes (e.g. outlines of other blobs)
    '''
    segs = np.argmax(outputs.cpu().detach().numpy(), axis=1) # FLATTEN TO [n_obs, n_pix, n_pix] masks by taking argmax along classes axis
    edge_pix = 0
    for batch in range(segs.shape[0]):
        cv2.imwrite(f"{temp_dir}/temp_img.png", segs[batch,:,:]) 
        edges = np.array(cv2.Canny(cv2.imread(f"{temp_dir}/temp_img.png"),0,1.5))
        try:
            edge_pix += len(edges[edges > 0])
        except TypeError as E:
            d = f"../{temp_dir}/temp_img.png"
            raise ValueError(f'Error {E}\nedges is {edges}\ntemp img saved at {d}\nos.getcwd {os.getcwd()}\nos.path.exists({d}) {os.path.exists(d)}')
    loss = edge_pix/(segs.shape[0]*segs.shape[1]**2)

    return loss


def class_size_loss(outputs):
    '''
    Metric describing 
    e.g. should encourage segmentations where two classes contain roughly similar N pixels, while the third contains significantly fewer
    '''
    
    preds = outputs.cpu().detach().numpy() # no argmax, maintain one-hot layers
    N0 = np.sum(preds[:,0,:,:])
    N1 = np.sum(preds[:,1,:,:])
    N2 = np.sum(preds[:,2,:,:])

    N_low = np.min([N0, N1, N2])# number in the smallest class
    N_high = np.max([N0, N1, N2]) # number in the largest class
    N_mid = [N for N in [N0, N1, N2] if N not in [N_low, N_high]][0]
    
    #R1 = N_high/N_mid # should be ~1
    R2 = N_low/np.mean([N_mid, N_high]) # should be << 1
    
    loss = R2 
    
    
def region_size_loss(outputs):
    '''
    Ideally, want to identify/label each blob, then look at the distribution of sizes for blobs of each class
        BP blobs are the ones that have the on average smallest size
        IGr blobs are the one(s) with the smallest total num blobs
    '''
    
    return loss

    
def continuity_loss(outputs):
    '''
    Metric describing the degree to which neighbouring pixels are the same class
    '''

    diff_h = torch.abs(outputs[:, :, 1:, :] - outputs[:, :, :-1, :])
    diff_v = torch.abs(outputs[:, :, 1:] - outputs[:, :, :-1])
    loss = torch.mean(diff_h) + torch.mean(diff_v)

    return loss


def soft_n_cut_loss(image, enc, img_size, debug=False):
    '''
    From https://github.com/AsWali/WNet/blob/master/utils
    NOTE: operates on [n_obs, n_class, n_pix, n_pix], but should each class layer be a mask or probs??? Currently probs, which *I think* matches Xia implementation
    '''
    if debug: print(f'\t  inside soft_n_cut_loss', flush=True)
    loss = []
    batch_size = image.shape[0]
    k = enc.shape[1]
    if debug: print(f'\t  calculated wieghts', flush=True)
    weights = calculate_weights(image, batch_size, img_size)
    if debug: print(f'\t  k {k}', flush=True)
    for i in range(0, k):
        l = soft_n_cut_loss_single_k(weights, enc[:, (i,), :, :], batch_size, img_size, debug=debug)
        if debug: print(f'\t    i {i}, computed l', flush=True)
        loss.append(l) # GETS STUCK HERE? 
    if debug: print(f'\t  losses appended to loss', flush=True)
    da = torch.stack(loss)
    if debug: print(f'\t  computed da', flush=True)
    out = torch.mean(k - torch.sum(da, dim=0))
    if debug: print(f'\t  computed out', flush=True)

    return out


def calculate_weights(input, batch_size, img_size=(64, 64), ox=4, radius=5 ,oi=10):
    '''
    From https://github.com/AsWali/WNet/blob/master/utils (for use in soft_n_cut_loss)
    '''
    channels = 1
    input = torch.tensor(input) # in case running on npy
    image = torch.mean(input, dim=1, keepdim=True)
    h, w = img_size
    p = radius
    image = F.pad(input=image, pad=(p, p, p, p), mode='constant', value=0)
    kh, kw = radius*2 + 1, radius*2 + 1
    dh, dw = 1, 1
    patches = image.unfold(2, kh, dh).unfold(3, kw, dw)
    patches = patches.contiguous().view(batch_size, channels, -1, kh, kw)
    patches = patches.permute(0, 2, 1, 3, 4)
    patches = patches.view(-1, channels, kh, kw)
    center_values = patches[:, :, radius, radius]
    center_values = center_values[:, :, None, None]
    center_values = center_values.expand(-1, -1, kh, kw)
    k_row = (torch.arange(1, kh + 1) - torch.arange(1, kh + 1)[radius]).expand(kh, kw)
    if torch.cuda.is_available():
        k_row = k_row.cuda()
    distance_weights = (k_row ** 2 + k_row.T**2)
    mask = distance_weights.le(radius)
    distance_weights = torch.exp(torch.div(-1*(distance_weights), ox**2))
    distance_weights = torch.mul(mask, distance_weights)
    patches = torch.exp(torch.div(-1*((patches - center_values)**2), oi**2))
    return torch.mul(patches, distance_weights)


def soft_n_cut_loss_single_k(weights, enc, batch_size, img_size, radius=5, debug=False):
    '''
    From https://github.com/AsWali/WNet/blob/master/utils (for use in soft_n_cut_loss)
    '''

    channels = 1
    enc = torch.tensor(enc) # in case running on npy
    h, w = img_size
    p = radius
    kh, kw = radius*2 + 1, radius*2 + 1
    dh, dw = 1, 1
    encoding = F.pad(input=enc, pad=(p, p, p, p), mode='constant', value=0)
    seg = encoding.unfold(2, kh, dh).unfold(3, kw, dw)
    seg = seg.contiguous().view(batch_size, channels, -1, kh, kw)
    seg = seg.permute(0, 2, 1, 3, 4)
    seg = seg.view(-1, channels, kh, kw)
    nom = weights * seg
    nominator = torch.sum(enc * torch.sum(nom, dim=(1,2,3)).reshape(batch_size, h, w), dim=(1,2,3))
    denominator = torch.sum(enc * torch.sum(weights, dim=(1,2,3)).reshape(batch_size, h, w), dim=(1,2,3))
    out = torch.div(nominator, denominator)

    return out


class multiclass_MSE_loss(nn.Module):
    '''
    DONT USE THIS - IT DOESNT SEEM TO BE WORKING 
    '''

    def __init__(self):
        super(multiclass_MSE_loss, self).__init__()

    def forward(self, outputs, targets, bp_weight, dm_weight=None):
        """
        Compute MSE between preds and targets for each layer (Ig, DM, ..etc). 
        Sum to get total, applying wieghting to small classes.
        NOTE: in this blog, they do a similar thing for image classification https://towardsdatascience.com/implementing-custom-loss-functions-in-pytorch-50739f9e0ee1
        NOTE: should really just generalize this to any num classes
        
        Parameters
        ----------
        outputs : `numpy.ndarray` of shape [n_obs, n_classes, n_pix, n_pix]
            Model outputs before application of activation function
        
        Returns
        -------
        targets : `numpy.ndarray` [n_obs, n_classes, n_pix, n_pix]
            One-hot encoded target values. Ordering along n_class axis must be the same as for preds.
        """

        probs = torch.sigmoid(outputs) # use sigmiod to turn into class probs
        # preds =  probs_to_preds(probs) # JUST ADDED - WILL IT FIX IT?
        preds = probs

        mse = nn.MSELoss()
        n_classes = len(targets[0,:,0,0])
        if n_classes == 3:
            mse_ig, mse_gr, mse_bp = 0, 0, 0
            for idx in range(probs.shape[0]): # loop through images in batch
                mse_ig += mse(targets[idx,0,:,:], preds[idx,0,:,:])
                mse_gr += mse(targets[idx,1,:,:], preds[idx,1,:,:])
                mse_bp += mse(targets[idx,2,:,:], preds[idx,2,:,:]) * bp_weight
            loss =  mse_ig + mse_gr + bp_weight*mse_bp
        if n_classes == 4:
            mse_ig, mse_dm, mse_gr, mse_bp = 0, 0, 0, 0
            for idx in range(preds.shape[0]): # loop through images in batch
                mse_ig += mse(targets[idx,0,:,:], preds[idx,0,:,:])
                mse_dm += mse(targets[idx,1,:,:], preds[idx,1,:,:]) * dm_weight
                mse_gr += mse(targets[idx,2,:,:], preds[idx,2,:,:])
                mse_bp += mse(targets[idx,3,:,:], preds[idx,3,:,:]) * bp_weight
            loss =  mse_ig + dm_weight*mse_dm + mse_gr + bp_weight*mse_bp
        return loss
    
    

class OpeningLoss2D(nn.Module):
    '''
    From Benoit's student's project https://github.com/tremblaybenoit/search/blob/main/src_freeze/loss.py 
    Computes the Mean Squared Error between computed class probabilities their grey opening.  
    Grey opening is a morphology operation, which performs an erosion followed by dilation.  
    Encourages the network to return sharper boundaries to objects in the class probabilities.
    '''

    def __init__(self, radius: int = 2):
        r"""
        :param radius: Radius for the channel-wise grey opening operation
        """
        super(OpeningLoss2D, self).__init__()
        self.radius = radius

    def forward(self, labels: torch.Tensor, *args) -> torch.Tensor:
        r"""Computes the Opening loss -- i.e. the MSE due to performing a greyscale opening operation.

        :param labels: Predicted class probabilities [NOTE: NOT MASKS!]
        :param args: Extra inputs, in case user also provides input/output image values.
        :return: Opening loss
        """
        smooth_labels = labels.clone().detach().cpu().numpy()
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                smooth_labels[i, j] = sndi.grey_opening(smooth_labels[i, j], self.radius)

        smooth_labels = torch.from_numpy(smooth_labels.astype(np.float32))
        if labels.device.type == 'cuda':
            smooth_labels = smooth_labels.cuda()

        return nn.MSELoss()(labels, smooth_labels.detach())
    

def MSE(pred, truth):
    mse = nn.MSELoss()
    return mse(pred, truth)
