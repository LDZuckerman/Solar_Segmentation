import numpy as np
import cv2
import sunpy
import scipy.ndimage as sndi
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
import astropy.io.fits as fits 
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import skimage as sk
import scipy.stats as stats
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader

######## Functions for NNs

# Dataset
class MyDataset(Dataset):

    def __init__(self, image_dir, mask_dir, norm=False, channels=[], n_classes=2, randomSharp=False, im_size=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.norm = norm
        self.channels = channels
        self.n_classes = n_classes
        self.randomSharp = randomSharp
        self.im_size = im_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Get image
        img_path = os.path.join(self.image_dir, self.images[index]) # path to one data image
        img = np.load(img_path).newbyteorder().byteswap() 
        if self.randomSharp: # add 50% chance of image being blurred/sharpened by a factor pulled from a skewed guassian (equal chance of 1/4 and 4)
            img =((img - np.nanmin(img))/(np.nanmax(img) - np.nanmin(img))) # first must [0, 1] normalize
            img = torch.from_numpy(np.expand_dims(img, axis=0)) # transforms expect a batch dimension
            n = stats.halfnorm.rvs(loc=1, scale=1, size=1)[0]
            s = n if np.random.rand(1)[0] < 0.5 else 1/n
            transf = transforms.RandomAdjustSharpness(sharpness_factor=s, p=0.5)
            img = transf(img)[0] # remove batch dimension for now
        if self.norm:  # normalize 
            img = (img - np.mean(img))/np.std(img) # normalize to std normal dist
        if self.channels != []: # Add feature layers
            image = np.zeros((len(self.channels)+1, img.shape[0], img.shape[1]), dtype=np.float32) # needs to be float32 not float64
            image[0, :, :] = img
            for i in range(len(self.channels)):
                image[i+1, :, :] = get_feature(img, self.channels[i], index)
        else: # Add dummy axis
            image = np.zeros((1, img.shape[0], img.shape[1]), dtype=np.float32) # needs to be float32 not float64
            image[0, :, :] = img
        if self.im_size != None: # cut to desired size, e.g. to make divisible by 2 5 times, for WNet
            image = image[:, 0:self.im_size, 0:self.im_size]
        # Get labels
        mask_path = os.path.join(self.mask_dir, 'SEG_'+self.images[index]) # path to one labels image THIS SHOULD ENSURE ITS THE MASK CORRESPONDING TO THE CORRECT IMAGE
        labels = np.load(mask_path).newbyteorder().byteswap() 
        if self.n_classes==4: # One-hot encode targets so they are the correct size
            mask = np.zeros((4, labels.shape[0], labels.shape[1]), dtype=np.float32) # needs to be float32 not float64
            mask_gr, mask_ig, mask_bp, mask_dm = np.zeros_like(labels), np.zeros_like(labels), np.zeros_like(labels), np.zeros_like(labels)
            mask_ig[labels == 0] = 1 # 1 where intergranule, 0 elsewhere
            mask_dm[labels == 0.5] = 1 # 1 where dim middle, 0 elsewhere
            mask_gr[labels == 1] = 1 # 1 where granule, 0 elsewhere
            mask_bp[labels == 1.5] = 1 # 1 where bright point, 0 elsewhere
            mask[0, :, :] = mask_ig
            mask[1, :, :] = mask_dm
            mask[2, :, :] = mask_gr
            mask[3, :, :] = mask_bp
        if self.n_classes==3: # One-hot encode targets so they are the correct size
            mask = np.zeros((3, labels.shape[0], labels.shape[1]), dtype=np.float32) # needs to be float32 not float64
            mask_gr, mask_ig, mask_bp = np.zeros_like(labels), np.zeros_like(labels), np.zeros_like(labels)
            mask_ig[labels == 0] = 1 # 1 where intergranule, 0 elsewhere
            mask_gr[labels == 1] = 1 # 1 where granule, 0 elsewhere
            mask_bp[labels == 1.5] = 1 # 1 where bright point, 0 elsewhere
            mask[0, :, :] = mask_ig
            mask[1, :, :] = mask_gr
            mask[2, :, :] = mask_bp
        elif self.n_classes==2: # One-hot encode targets so they are the correct size
            mask = np.zeros((2, labels.shape[0], labels.shape[1]), dtype=np.float32) # needs to be float32 not float64
            mask_gr, mask_ig = np.zeros_like(labels), np.zeros_like(labels)
            mask_ig[labels == 0] = 1 # 1 where intergranule, 0 elsewhere
            mask_gr[labels == 1] = 1 # 1 where granule, 0 elsewhere
            mask[0, :, :] = mask_ig
            mask[1, :, :] = mask_gr
        if self.im_size != None: # cut to desired size, e.g. to make divisible by 2 4 times, for WNet
            mask = mask[:, 0:self.im_size, 0:self.im_size]
        return image, mask

def probs_to_preds(probs):
    '''
    Helper function to turn 3D class probs into 2D arrays of predictions (also changes tensor to numpy)
    '''
    if probs.shape[1] == 1: # if binary (predictions have 1 layer)
        preds = (probs > 0.5).float() # turn regression value (between zero and 1, e.g. "prob of being 1") into predicted 0s and 1s
    else: # if muliclasss (predictions have n_classes layers)
        preds = np.argmax(probs.detach().numpy(), axis=1) # turn probabilities [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]

    return preds

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

class multiclass_MSE_loss(nn.Module):

    def __init__(self):
        super(multiclass_MSE_loss, self).__init__()

    def forward(self, outputs, targets, dm_weight=1, bp_weight=1):
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
        preds =  probs_to_preds(probs) # JUST ADDED - WILL IT FIX IT?
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

def check_inputs(train_ds, train_loader, savefig=False, name=None):

    # Check data is loaded correctly
    print('Train data:')
    print(f'     {len(train_ds)} obs, broken into {len(train_loader)} batches')
    train_features, train_labels = next(iter(train_loader))
    shape = train_features.size()
    print(f'     Each batch has data of shape {train_features.size()}, e.g. {shape[0]} images, {[shape[2], shape[3]]} pixels each, {shape[1]} layers (features)')
    shape = train_labels.size()
    print(f'     Each batch has labels of shape {train_labels.size()}, e.g. {shape[0]} images, {[shape[2], shape[3]]} pixels each, {shape[1]} layers (classes)')
    if savefig:
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(2*4, 4*4)); axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
        ax1.set_title('image')
        ax2.set_title('labels')
        for i in range(0, 8, 2):
            X, y = next(iter(train_loader))
            y = onehot_to_map(y)
            im1 = axs[i].imshow(X[0,0,:,:]); plt.colorbar(im1, ax=axs[i]) # first img in batch, first channel
            im2 = axs[i+1].imshow(y[0,:,:]); plt.colorbar(im2, ax=axs[i+1]) # first y in batch, already class-collapsed
        plt.savefig(f'traindata_{name}')


def get_feature(img, name, index):

    if name == 'gradx': a = np.gradient(img)[0]
    elif name == 'grady': a = np.gradient(img)[1]
    elif name == 'squared': a = img**2
    elif 'power' in name:
        n = int(name[-1])
        a = img**n
    # elif name == 'deltaBinImg':
    #     UNet1seg = np.load(f'../UNet1_outputs/pred_{index}.npy') # zeros and ones
    #     imgnorm = (img - np.mean(img))/np.std(img) # normalize to range [0, 1]
    #     a = UNet1seg - imgnorm # difference between binary segmentation and image (pos for bp, neg for dm)
    else: raise ValueError(f'Channel name {name} not recognized')

    return a

def compute_validation_results(output_dir, n_classes=2):
    '''
    Compute the total percent correct, and percent correct on each class, using outputs of 
    NN on validation data
    '''

    truefiles = [file for file in os.listdir(output_dir) if 'true' in file]
    predfiles = [file for file in os.listdir(output_dir) if 'pred' in file]

    pix_correct, ig_correct, dm_correct, gr_correct, bp_correct = 0, 0, 0, 0, 0
    tot_pix, tot_ig, tot_dm, tot_gr, tot_bp = 0, 0, 0, 0, 0

    for i in range(len(truefiles)):
        true = np.load(f'{output_dir}/{truefiles[i]}')
        preds = np.load(f'{output_dir}/{predfiles[i]}')

        # if len(np.unique(true)) == 3:
        #     plt.figure(); im = plt.imshow(true); plt.colorbar(im)
        #     print()

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

    # if n_classes == 2:
    #     for i in range(len(truefiles)):
    #         true = np.load(f'{output_dir}/{truefiles[i]}')
    #         preds = np.load(f'{output_dir}/{predfiles[i]}')
    #         pix_correct += len(np.where(preds.flatten() == true.flatten())[0]) #(preds == true).sum()
    #         tot_pix += len(preds.flatten()) # torch.numel(preds)
    #         ig_correct += len(np.where((preds.flatten() == true.flatten()) & (true.flatten() == 0))[0])
    #         tot_ig += len(np.where(true.flatten() == 0)[0])
    #         gr_correct += len(np.where((preds.flatten() == true.flatten()) & (true.flatten() == 1))[0])
    #         tot_gr += len(np.where(true.flatten() == 1)[0])
    #     pct_correct = pix_correct/tot_pix*100
    #     pct_ig_correct = ig_correct/tot_ig*100
    #     pct_dm_correct = np.NaN
    #     pct_gr_correct = gr_correct/tot_gr*100
    #     pct_bp_correct = np.NaN
    # elif n_classes == 3:
    #     for i in range(len(truefiles)):
    #         true = np.load(f'{output_dir}/{truefiles[i]}')
    #         preds = np.load(f'{output_dir}/{predfiles[i]}')
    #         pix_correct += len(np.where(preds.flatten() == true.flatten())[0]) #(preds == true).sum()
    #         tot_pix += len(preds.flatten()) # torch.numel(preds)
    #         ig_correct += len(np.where((preds.flatten() == true.flatten()) & (true.flatten() == 0))[0])
    #         tot_ig += len(np.where(true.flatten() == 0)[0])
    #         gr_correct += len(np.where((preds.flatten() == true.flatten()) & (true.flatten() == 1))[0])
    #         tot_gr += len(np.where(true.flatten() == 1)[0])
    #         bp_correct += len(np.where((preds.flatten() == true.flatten()) & (true.flatten() == 1.5))[0])
    #         tot_bp += len(np.where(true.flatten() == 1.5)[0])
    #     pct_correct = pix_correct/tot_pix*100
    #     pct_ig_correct = ig_correct/tot_ig*100
    #     pct_dm_correct = np.NaN
    #     pct_gr_correct = gr_correct/tot_gr*100
    #     pct_bp_correct = bp_correct/tot_bp*100
    # elif n_classes == 4:
    #     for i in range(len(truefiles)):
    #         true = np.load(f'{output_dir}/{truefiles[i]}')
    #         preds = np.load(f'{output_dir}/{predfiles[i]}')
    #         fig, axs = plt.subplots(1,2)
    #         axs[0].imshow(true); axs[1].imshow(preds)
    #         if i ==5: a=b
    #         pix_correct += len(np.where(preds.flatten() == true.flatten())[0]) #(preds == true).sum()
    #         tot_pix += len(preds.flatten()) # torch.numel(preds)
    #         ig_correct += len(np.where((preds.flatten() == true.flatten()) & (true.flatten() == 0))[0])
    #         tot_ig += len(np.where(true.flatten() == 0)[0])
    #         dm_correct += len(np.where((preds.flatten() == true.flatten()) & (true.flatten() == 0.5))[0])
    #         tot_dm += len(np.where(true.flatten() == 0.5)[0])
    #         gr_correct += len(np.where((preds.flatten() == true.flatten()) & (true.flatten() == 1))[0])
    #         tot_gr += len(np.where(true.flatten() == 1)[0])
    #         bp_correct += len(np.where((preds.flatten() == true.flatten()) & (true.flatten() == 1.5))[0])
    #         tot_bp += len(np.where(true.flatten() == 1.5)[0])
    #     pct_correct = pix_correct/tot_pix*100
    #     pct_ig_correct = ig_correct/tot_ig*100
    #     pct_dm_correct = dm_correct/tot_dm*100
    #     pct_gr_correct = gr_correct/tot_gr*100
    #     pct_bp_correct = bp_correct/tot_bp*100

    return pct_correct, pct_ig_correct, pct_dm_correct, pct_gr_correct, pct_bp_correct

# From https://github.com/AsWali/WNet/blob/master/
def calculate_weights(input, batch_size, img_size=(64, 64), ox=4, radius=5 ,oi=10):
    channels = 1
    image = torch.mean(input, dim=1, keepdim=True)
    h, w = img_size
    p = radius
    image = F.pad(input=image, pad=(p, p, p, p), mode='constant', value=0)
    # Use this to generate random values for the padding.
    # randomized_inputs = (0 - 255) * torch.rand(image.shape).cuda() + 255
    # mask = image.eq(0)
    # image = image + (mask *randomized_inputs)
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

# From https://github.com/AsWali/WNet/blob/master/
def soft_n_cut_loss_single_k(weights, enc, batch_size, img_size, radius=5):
    channels = 1
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
    return torch.div(nominator, denominator)

# From https://github.com/AsWali/WNet/blob/master/
def soft_n_cut_loss(image, enc, img_size):
    loss = []
    batch_size = image.shape[0]
    k = enc.shape[1]
    weights = calculate_weights(image, batch_size, img_size)
    for i in range(0, k):
        loss.append(soft_n_cut_loss_single_k(weights, enc[:, (i,), :, :], batch_size, img_size))
    da = torch.stack(loss)
    return torch.mean(k - torch.sum(da, dim=0))

# From https://github.com/taoroalin/WNet/blob/master/train.py
def gradient_regularization(softmax):
    # THIS SEEMS TO BE EXPECTING CHANNELS DIM BEFORE BATCH DIM????
    vertical_sobel=torch.nn.Parameter(torch.from_numpy(np.array([[[[1,  0,  -1], 
                                            [1,  0,  -1], 
                                            [1,  0,  -1]]]])).float(), requires_grad=False)

    horizontal_sobel=torch.nn.Parameter(torch.from_numpy(np.array([[[[1,   1,  1], 
                                              [0,   0,  0], 
                                              [-1 ,-1, -1]]]])).float(), requires_grad=False)
    print(f'softmax: {softmax.shape}')
    print(f'softmax.shape[0]: {softmax.shape[0]}')
    for i in range(softmax.shape[0]):
        print(f'i {i}')
        print(f'\tsoftmax[:, i]: {softmax[:, i].shape}')
    conv = [F.conv2d(softmax[:, i].unsqueeze(1), vertical_sobel) for i in range(softmax.shape[0])]
    print(f'conv: {np.array(conv).shape}')


    vert=torch.cat([F.conv2d(softmax[:, i].unsqueeze(1), vertical_sobel) for i in range(softmax.shape[0])], 1)
    hori=torch.cat([F.conv2d(softmax[:, i].unsqueeze(1), horizontal_sobel) for i in range(softmax.shape[0])], 1)
    print('vert', torch.sum(vert))
    print('hori', torch.sum(hori))
    mag=torch.pow(torch.pow(vert, 2)+torch.pow(hori, 2), 0.5)
    mean=torch.mean(mag)
    return mean

# From Benoit's student's project https://github.com/tremblaybenoit/search/blob/main/src_freeze/loss.py 
class OpeningLoss2D(nn.Module):
    r"""Computes the Mean Squared Error between computed class probabilities their grey opening.  Grey opening is a
    morphology operation, which performs an erosion followed by dilation.  Conceptually, this encourages the network
    to return sharper boundaries to objects in the class probabilities.

    NOTE:  Original loss term -- not derived from the paper for NCutLoss2D."""

    def __init__(self, radius: int = 2):
        r"""
        :param radius: Radius for the channel-wise grey opening operation
        """
        super(OpeningLoss2D, self).__init__()
        self.radius = radius

    def forward(self, labels: torch.Tensor, *args) -> torch.Tensor:
        r"""Computes the Opening loss -- i.e. the MSE due to performing a greyscale opening operation.

        :param labels: Predicted class probabilities
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


######## Preprocessing 

def histogram_equalization(data, n_bins=256):

   #get image histogram
   hist, bins = np.histogram(data.flatten(), n_bins, normed=True)
   cdf = hist.cumsum()
   cdf = 255 * cdf / cdf[-1] #normalize
   # use linear interpolation of cdf to find new pixel values
   data_out = np.interp(data.flatten(), bins[:-1], cdf)
   data_out = data_out.reshape(data.shape)

   return data_out

def get_localRMS(data):
    # Should I do this?
    return None

def match_to_firstlight(obs_data, n_bins=2000):

    # Read in "good" image and normalize
    synth_data = fits.open('../Data/DKIST_example.fits')[0].data
    synth_data = (synth_data-np.nanmean(synth_data))/np.nanstd(synth_data)
    # Compute good image histogram and cdf
    synth_bins = np.linspace(np.nanmin(synth_data), np.nanmax(synth_data), n_bins)
    width_synth_bins = synth_bins[1]-synth_bins[0]
    synth_hist, _ = np.histogram(synth_data[np.isfinite(synth_data)].flatten(), bins=synth_bins, density=True)
    synth_hist = np.expand_dims(synth_hist, axis=0) # since not loading data as stacked array
    synth_cdf = np.cumsum(np.nanmean(synth_hist*width_synth_bins, axis=0))
    # Normalize "bad" image
    obs_data = (obs_data-np.nanmean(obs_data))/np.nanstd(obs_data)
    # Compute good image histogram and cdf
    obs_bins = np.linspace(np.nanmin(obs_data), np.nanmax(obs_data), n_bins)
    width_obs_bins = obs_bins[1]-obs_bins[0]
    obs_hist, _ = np.histogram(obs_data[np.isfinite(obs_data)].flatten(), bins=obs_bins, density=True)
    obs_hist = np.expand_dims(obs_hist, axis=0) # since not loading data as stacked array
    obs_cdf = np.cumsum(np.nanmean(obs_hist*width_obs_bins, axis=0))
    # Perform matching
    obs_data_matched = hist_matching(obs_data, obs_cdf, obs_bins, synth_cdf, synth_bins)
    obs_data_matched = obs_data_matched.reshape(np.shape(obs_data))
    return obs_data_matched

    
def hist_matching(data_in, cdf_in, bins_in, cdf_out, bins_out):

    bins_in = 0.5*(bins_in[:-1] + bins_in[1:]) # Points for interpolation (input bins contain the edges)
    bins_out = 0.5*(bins_out[:-1] + bins_out[1:])
    cdf_tmp = np.interp(data_in.flatten(), bins_in.flatten(), cdf_in.flatten())  # Interpolation
    data_out = np.interp(cdf_tmp, cdf_out.flatten(), bins_out.flatten())
    return data_out

######## Functions for the machine learning segmentation of various solar features

def add_kernel_feats(df, dataflat):

    df_new = df.copy()
    k1 = np.array([[1, 1, 1],  # blur (maybe useful?)
                [1, 1, 1],
                [1, 1, 1]])/9
    k2 = np.array([[0, -1, 0],  # sharpening (probably not useful?)
                [-1, 5, -1],
                [0, -1, 0]])
    k3 = np.array([[-1, -1, -1],  # edge detection (probably not useful?)
                [-1, 8, -1],
                [-1, -1, -1]])
    kernels = [k1, k2, k3]
    for i in range(len(kernels)):
        kernel = kernels[i]
        filtered_img = cv2.filter2D(dataflat, -1, kernel).reshape(-1) # filtered_img = cv2.GaussianBlur(data, (5,5), 0)#.reshape(-1)
        # fig, (ax1, ax2) = plt.subplots(1,2); ax1.imshow(data); ax2.imshow(filtered_img)
        df_new['kernel'+str(i)] = filtered_img
    return df_new

def add_gradient_feats(df, data):

    # Attempted to use cv2.Laplacian, but require dtype uint8, and converting caused issues with the normalization #cv2.Laplacian(data,cv2.CV_64F).reshape(-1)
    df_new = df.copy()
    df_new['gradienty'] = np.gradient(data)[0].reshape(-1)
    df_new['gradientx'] = np.gradient(data)[1].reshape(-1)
    return df_new

def add_sharpening_feats(df, dataflat):

    df_new = df.copy()
    df_new['value2'] = dataflat**2
    return df_new

def pre_proccess(data, labels, gradientFeats=True, kernalFeat=True):

    # Flatten features and labels
    dataflat = data.reshape(-1)
    labelsflat = labels.reshape(-1)
    # Put features and labels into df
    df = pd.DataFrame()
    df['OG_value'] = dataflat
    df = add_kernel_feats(df, dataflat) # Add values of different filters as features
    df = add_gradient_feats(df, data) # Add value of gradient as feature
    df['labels'] =  labelsflat
    # Make X and Y
    X =  df.drop(labels =["labels"], axis=1)
    Y = df['labels']
    Y = preprocessing.LabelEncoder().fit_transform(Y) # turn floats 0, 1, to categorical 0, 1
    return X, Y

def post_process(preds, data):

    preds = np.copy(preds).astype(float)  # Float conversion for correct region numbering.
    preds2 = np.ones_like(preds)*20 # Just to aviod issues later on 
    # If its a 2-value seg
    if len(np.unique(preds)) == 2:
        # Assign a number to each predicted region
        labeled_preds = skimage.measure.label(preds + 1, connectivity=2)
        values = np.unique(labeled_preds)
        # Find numbering of the largest region (IG region)
        size = 0
        for value in values:
            if len((labeled_preds[labeled_preds == value])) > size:
                IG_value = value
                size = len(labeled_preds[labeled_preds == value])
        # Where labeled_preds=IG_value set preds2 to zero, otherwise 1
        preds2[labeled_preds == IG_value] = 0
        preds2[labeled_preds != IG_value] = 1 
    # If its a 3-value seg
    elif len(np.unique(preds)) == 3:
        # WAIT BUT THIS WONT HELP ANYTHING CUASE NONE OF THE 3-VALUE ONES ID BPSs
        # NEED AN ALGORITHM TO MERGE N CLUSTERS INTO 3
        # 
        # Find the seg value of the region corresponding to the lowest and highest avg pix value in the og data 
        highest_mean = 0
        lowest_mean = sum(data) # will never be higher than this
        for seg_val in np.unique(preds):
            if np.mean(data[preds == seg_val]) < lowest_mean:
                IG_value = seg_value
            if np.mean(data[preds == seg_val]) > highest_mean:
                BP_value = seg_value
        # Where labeled_preds=IG_value set preds2 to zero, where BP_value, 0.5, and else (granule), 1
        preds2[labeled_preds == IG_value] = 0
        preds2[labeled_preds == BP_value] = 0.5
        preds2[labeled_preds != IG_value and labeled_preds != BP_value] 
    else:
        print('NOT YET IMPLEMENTED: NEED TO FIND A WAY TO GET >3-VALUE SEG INTO FORM COMPARABLE TO LABELS'); a=b

    return preds2

def eval_metrics(metric, true, pred):
    """
    Ways of evaluating extent to which true and preds are the same, BUT ONLY IN THE CASE THAT THEY SHOULD BE.
    E.g. for comparing labels to outputs of surpervised method, or to outputs of 2-value unsupervised method
    where those two values have been converted to 0 = IG, 1 = G.
    None of these will be usefull for >2-value unsupervised methods untill I figure out how to algorithmically
    force the outputs (including combining groups) into IG, G, BP, DM values. 
    """
    if metric == 'pct_correct':
        # Could be ok for our use; in gernal bad if huge class imbalance (could get high score by predicting all one class)
        return len(np.where(preds==true)[0])/len(preds)
    if metric == 'accuracy_score':
        # Avg of the area of overlap over area of union for each class (like Jaccard score but for two or more classes)
        return metrics.accuracy_score(true, pred)

######## Functions for use with UNet tutorial 2

import torch
import torchvision

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval() # set model into eval mode
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"saved_images/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
    model.train() # set model back into train mode


######## Functions from DKISTSegmentation project for validation of ML methods ###########

def segment_array_v2(map, resolution, *, skimage_method="li", mark_dim_centers=False, mark_BP=True, bp_min_flux=None, bp_max_size=0.15, footprint=250):
    
    """
    NOT EXACTLY THE SAME AS SUNKIT-IMAGE VERSION (DIFFERENT CLASS LABELS, NPY VS MAP, ADDS MARK_FAC FLAG)
    Segment an optical image of the solar photosphere into four-value maps with:

     * 0 as intergranule
     * 0.5 as "dim-middle" (optional)
     * 1 as granule
     * 1.5 "brightpoint" (optional)

    Parameters
    ----------
    smap : `numpy.ndarray`
        NumPy array containing data to segment.
    resolution : `float`
        Spatial resolution (arcsec/pixel) of the data.
    skimage_method : {"li", "otsu", "isodata", "mean", "minimum", "yen", "triangle"}
        scikit-image thresholding method, defaults to "li".
        Depending on input data, one or more of these methods may be
        significantly better or worse than the others. Typically, 'li', 'otsu',
        'mean', and 'isodata' are good choices, 'yen' and 'triangle' over-
        identify intergranule material, and 'minimum' over identifies granules.
    mark_dim_centers : `bool`
        Whether to mark dim granule centers as a separate category for future exploration.
    mark_bright_points : `bool`
        Whether to mark bright points as a seperate catagory for future exploration
    bp_min_flux : `float`
        Minimum flux level per pixel for a region to be considered a Bright Point.
        Defaalt is one standard deviation above the mean flux.
    bp_max_size: `float`
        Maximum diameter (arcsec) to consider a region a Bright Point.
        Defualt of 0.15. 

    Returns
    -------
    segmented_map : `numpy.ndarray`
        NumPy array containing a segmented image (with the original header).
    """

    # if skimage_method not in METHODS:
    #     raise TypeError("Method must be one of: " + ", ".join(METHODS))

    # Obtain local histogram equalization of map.
    map_norm = ((map - np.nanmin(map))/(np.nanmax(map) - np.nanmin(map))) * 225 # min-max normalization to [0, 225] 
    map_HE = sk.filters.rank.equalize(map_norm.astype(int), footprint=sk.morphology.disk(footprint)) # MAKE FOOTPRINT SIZE DEPEND ON RESOLUTION!!!
    # Apply initial skimage threshold for initial rough segmentation into granules and intergranules.
    median_filtered = sndi.median_filter(map_HE, size=3)
    threshold = get_threshold_v2(median_filtered, skimage_method)
    segmented_image = np.uint8(median_filtered > threshold)
    # Fix the extra intergranule material bits in the middle of granules.
    seg_im_fixed = trim_intergranules_v2(segmented_image, mark=mark_dim_centers)
    # Mark faculae and get final granule and facule count.
    if mark_BP: seg_im_markfac, faculae_count, granule_count = mark_faculae_v2b(seg_im_fixed, map, map_HE, resolution, bp_min_flux, bp_max_size)
    else: seg_im_markfac = seg_im_fixed
    # logging.info(f"Segmentation has identified {granule_count} granules and {faculae_count} faculae")
    segmented_map = seg_im_markfac
    return segmented_map

def get_threshold_v2(data, method):
    """
    Get the threshold value using given skimage segmentation type.

    Parameters
    ----------
    data : `numpy.ndarray`
        Data to threshold.
    method : {"li", "otsu", "isodata", "mean", "minimum", "yen", "triangle"}
        scikit-image thresholding method.

    Returns
    -------
    threshold : `float`
        Threshold value.
    """
    
    if len(data.flatten()) > 2000**2:
        data = np.random.choice(data.flatten(), (500, 500))
        print(f'\tWARNING: data too big so computing threshold based on random samples reshaped to 500x500 image')

    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be an instance of a np.ndarray")
    elif method == "li":
        threshold = skimage.filters.threshold_li(data)
    if method == "otsu":
        threshold = skimage.filters.threshold_otsu(data)
    elif method == "yen":
        threshold = skimage.filters.threshold_yen(data)
    elif method == "mean":
        threshold = skimage.filters.threshold_mean(data)
    elif method == "minimum":
        threshold = skimage.filters.threshold_minimum(data)
    elif method == "triangle":
        threshold = skimage.filters.threshold_triangle(data)
    elif method == "isodata":
        threshold = skimage.filters.threshold_isodata(data)
    # else:
    #     raise ValueError("Method must be one of: " + ", ".join(METHODS))
    return threshold

def trim_intergranules_v2(segmented_image, mark=False):
    """
    Remove the erroneous identification of intergranule material in the middle
    of granules that the pure threshold segmentation produces.

    Parameters
    ----------
    segmented_image : `numpy.ndarray`
        The segmented image containing incorrect extra intergranules.
        Will have small padding of zeros around edges.
    mark : `bool`
        If `False` (the default), remove erroneous intergranules.
        If `True`, mark them as 0.5 instead (for later examination).

    Returns
    -------
    segmented_image_fixed : `numpy.ndarray`
        The segmented image without incorrect extra intergranules.
    """

    if len(np.unique(segmented_image)) > 2:
        raise ValueError("segmented_image must only have values of 1 and 0.")
    segmented_image_fixed = np.copy(segmented_image).astype(float)  # Float conversion for correct region labeling.
    # Add padding of IG around edges, because if edges are all GR, will ID all DM as IG
    pad = int(np.shape(segmented_image)[0]/200)
    segmented_image_fixed[:,0:pad] = 0 
    segmented_image_fixed[0:pad,:] = 0 
    segmented_image_fixed[:,-pad:] = 0 
    segmented_image_fixed[-pad:,:] = 0 
    labeled_seg = skimage.measure.label(segmented_image_fixed + 1, connectivity=2)
    values = np.unique(labeled_seg) 
    # Find value of the large continuous 0-valued region.
    size = 0
    print(f'\tloop 1 to {len(values)} (should take like 2 minutes)')
    for value in values:
        if len((labeled_seg[labeled_seg == value])) > size: # if bigger than previous largest
            if sum(segmented_image[labeled_seg == value] == 0): # if a zero (IG) region
                real_IG_value = value
                size = len(labeled_seg[labeled_seg == value])
    # Set all other 0 regions to mark value (1 or 0.5).
    print(f'\tloop 2 to {len(values)} (should take like 2 minutes)')
    for value in values:
        if np.sum(segmented_image[labeled_seg == value]) == 0:
            if value != real_IG_value:
                if not mark:
                    segmented_image_fixed[labeled_seg == value] = 1
                elif mark:
                    segmented_image_fixed[labeled_seg == value] = 0.5
    
    # segmented_image_return = np.copy(segmented_image)
    # segmented_image_return[:,0:pad] = segmented_image_fixed[:,0:pad]
    # segmented_image_return[0:pad,:] = segmented_image_fixed[0:pad,:] 
    # segmented_image_return[:,-pad:] = segmented_image_fixed[:,-pad:]
    # segmented_image_return[-pad:,:] = segmented_image_fixed[-pad:,:] 
    # plt.figure(); im = plt.imshow(segmented_image_fixed, origin='lower'); plt.colorbar(im); plt.title('segmented_image_fixed')
    # segmented_image_return = np.copy(segmented_image)
    # fixed = segmented_image_fixed[pad:-pad, pad:-pad] # np.zeros_like(segmented_image_fixed[pad:-pad, pad:-pad])  #
    # plt.figure(); im = plt.imshow(segmented_image_return, origin='lower'); plt.colorbar(im); plt.title('segmented_image_return')
    # plt.figure(); im = plt.imshow(fixed, origin='lower'); plt.colorbar(im); plt.title('fixed')
    # segmented_image_return[pad:-pad, pad:-pad] = fixed # np.random.normal(size=segmented_image_fixed[pad:-pad, pad:-pad].shape) # segmented_image_fixed[pad:-pad, pad:-pad] 
    # plt.figure(); im = plt.imshow(segmented_image_return, origin='lower'); plt.colorbar(im); plt.title('segmented_image_return')
    
    return segmented_image_fixed

def mark_faculae_v2b(segmented_image, data, HE_data, resolution, bp_min_flux=None, bp_max_size=0.15):
    """
    Mark faculae separately from granules - give them a value of 1.5 not 1.

    Parameters
    ----------
    segmented_image : `numpy.ndarray`
        The segmented image containing incorrect middles.
    data : `numpy array`
        The original image (not normalized or equalized)
    resolution : `float`
        Spatial resolution (arcsec/pixel) of the data.
    bp_min_flux : `float`
        Minimum flux level per pixel for a region to be considered a Bright Point.
        Default is 1 standard deviation above the mean flux (using equalized data)
    bp_max_size : `float`
        Maximum diameter (arcsec) for a region to be considered a Bright Point.
        Defualt of 0.15. 

    Returns
    -------
    segmented_image_fixed : `numpy.ndrray`
        The segmented image with faculae marked as 1.5.
    faculae_count: `int`
        The number of faculae identified in the image.
    granule_count: `int`
        The number of granules identified, after re-classifcation of faculae.
    """

    # Check inputs and initialize output map
    if len(np.unique(segmented_image)) > 3:
        raise ValueError("segmented_image must have only values of 1, 0 and a 0.5 (if dim centers marked)")
    segmented_image_fixed = np.copy(segmented_image.astype(float))
    # Set BP pixel limit
    fac_pix_limit = (bp_max_size / resolution)**2 # Max area in pixels
    # Use equalized map to set BP flux threshold and extract thresholded map
    if bp_min_flux == None: 
        bp_min_flux = np.nanmean(HE_data) + 1.25*np.nanstd(HE_data) # General flux limit determined by visual inspection.
    bright_dim_seg = np.zeros_like(data)
    bright_dim_seg[HE_data > bp_min_flux] = 1
    # Label the bright regions and get list of values
    labeled_bright_dim_seg = skimage.measure.label(bright_dim_seg + 1, connectivity=2)
    values = np.unique(labeled_bright_dim_seg)

    # Obtain gradient map and set threshold for gradient on BP edges
    grad = np.abs(np.gradient(data)[0] + np.gradient(data)[1])
    bp_min_grad = np.quantile(grad, 0.95)

    # Loop through bright regions, select those under pixel limit and containing high gradient
    fac_count = 0
    print(f'\tloop 3 to {len(values)} (this is the sticking point)')
    for value in values:
         print(f'\t\t{value}', end='\r')
         if (bright_dim_seg[labeled_bright_dim_seg==value])[0]==1: # Check region is not the non-bp region
            # check that region is small.
            region_size = len(labeled_bright_dim_seg[labeled_bright_dim_seg==value])
            if region_size < fac_pix_limit:
                # check that region has high average gradient (maybe try max gradient?)
                region_mean_grad = np.mean(grad[labeled_bright_dim_seg==value])
                if region_mean_grad > bp_min_grad:
                    segmented_image_fixed[labeled_bright_dim_seg==value] = 1.5
                    fac_count += 1
    gran_count = len(values) - 1 - fac_count  # Subtract 1 for IG region.

    return segmented_image_fixed, fac_count, gran_count

def mark_faculae_v2(segmented_image, data, resolution, bp_min_flux=None, bp_max_size=0.15):
    """
    Mark faculae separately from granules - give them a value of 1.5 not 1.

    Parameters
    ----------
    segmented_image : `numpy.ndarray`
        The segmented image containing incorrect middles.
    data : `numpy array`
        The original image (not normalized or equalized)
    resolution : `float`
        Spatial resolution (arcsec/pixel) of the data.
    bp_min_flux : `float`
        Minimum flux level per pixel for a region to be considered a Bright Point.
        Defaalt is 0.25 standard deviations above the mean flux.
    bp_max_size : `float`
        Maximum diameter (arcsec) for a region to be considered a Bright Point.
        Defualt of 0.15. 

    Returns
    -------
    segmented_image_fixed : `numpy.ndrray`
        The segmented image with faculae marked as 1.5.
    faculae_count: `int`
        The number of faculae identified in the image.
    granule_count: `int`
        The number of granules identified, after re-classifcation of faculae.
    """
    
    fac_pix_limit = (bp_max_size / resolution)**2 # Max area in pixels
    if bp_min_flux == None: 
        bp_min_flux = np.nanmean(data) + 0.25 * np.nanstd(data) # General flux limit determined by visual inspection.
    if len(np.unique(segmented_image)) > 3:
        raise ValueError("segmented_image must have only values of 1, 0 and a 0.5 (if dim centers marked)")
    segmented_image_fixed = np.copy(segmented_image.astype(float))
    labeled_seg = skimage.measure.label(segmented_image + 1, connectivity=2)

    values = np.unique(labeled_seg)
    fac_count = 0
    print(f'\tloop 3 to {len(values)} (this is the sticking point)')
    small_regions = np.zeros_like(segmented_image)
    for value in values:
        print(f'\t\t{value}', end='\r')
        mask = np.zeros_like(segmented_image)
        mask[labeled_seg == value] = 1
        # Check that is a 1 (white) region.
        if np.sum(np.multiply(mask, segmented_image)) > 0:
            region_size = len(segmented_image_fixed[mask == 1])
            # check that region is small.
            if region_size < fac_pix_limit:
                small_regions[mask == 1] = 1
                # Check that peak flux very high.
                tot_flux = np.nansum(data[mask == 1])
                if np.max(data[mask == 1]) > bp_min_flux: # if tot_flux / region_size > bp_min_flux:
                    segmented_image_fixed[mask == 1] = 1.5
                    fac_count += 1
                    small_regions[mask == 1] = 2
    gran_count = len(values) - 1 - fac_count  # Subtract 1 for IG region.

    # im = plt.imshow(small_regions, origin='lower'); plt.colorbar(im)
    # plt.title('small regions (1 = bright enough)')
    # plt.savefig('test0625b')
    # # plt.hist(data.flatten(), bins=20, label='all pixels')
    # # plt.hist(all_region_maxfluxes, label='all region maximums')
    # # plt.hist(small_region_maxfluxes, label='small region maximums')
    # # plt.legend()
    # # plt.vlines([np.nanmean(data)+0.25*np.nanstd(data)], ymin=0, ymax=7e6, color='black', label='mean+0.25*SD')
    # # plt.legend()
    # # plt.yscale('log')
    # # plt.title('VBI_04_05_4096')
    # # plt.savefig('test0625')

    return segmented_image_fixed, fac_count, gran_count


def segment_array(map, resolution, *, skimage_method="li", mark_dim_centers=False, mark_BP=True, fac_brightness_limit=None, fac_pix_limit=None):
    """
    IDENTICAL TO FUNCTIONS IN DKISTSegmentation REPO AND Sunkit-Image PACKAGE.
    EXCEPT RETURNING ARRAY NOT MAP AND RE-ADDEDITION OF MARK_FAC FLAG

    Segment an optical image of the solar photosphere into four-value maps with:

     * 0 as intergranule
     * 0.5 as "dim-middle"
     * 1 as granule
     * 1.5 "brightpoint"

    Parameters
    ----------
    smap : `numpy.ndarray`
        NumPy array containing data to segment.
    resolution : `float`
        Spatial resolution (arcsec/pixel) of the data.
    skimage_method : {"li", "otsu", "isodata", "mean", "minimum", "yen", "triangle"}
        scikit-image thresholding method, defaults to "li".
        Depending on input data, one or more of these methods may be
        significantly better or worse than the others. Typically, 'li', 'otsu',
        'mean', and 'isodata' are good choices, 'yen' and 'triangle' over-
        identify intergranule material, and 'minimum' over identifies granules.
    mark_dim_centers : `bool`
        Whether to mark dim granule centers as a separate category for future exploration.

    Returns
    -------
    segmented_map : `numpy.ndarray`
        NumPy array containing a segmented image (with the original header).
    """

    # if skimage_method not in METHODS:
    #     raise TypeError("Method must be one of: " + ", ".join(METHODS))

    median_filtered = sndi.median_filter(map, size=3)
    # Apply initial skimage threshold.
    threshold = get_threshold(median_filtered, skimage_method)
    segmented_image = np.uint8(median_filtered > threshold)
    # Fix the extra intergranule material bits in the middle of granules.
    seg_im_fixed = trim_intergranules(segmented_image, mark=mark_dim_centers)
    # Mark faculae and get final granule and facule count.
    if mark_BP: seg_im_markfac, faculae_count, granule_count = mark_faculae(seg_im_fixed, map, resolution)
    else: seg_im_markfac = seg_im_fixed
    # logging.info(f"Segmentation has identified {granule_count} granules and {faculae_count} faculae")
    segmented_map = seg_im_markfac
    return segmented_map


def segment(smap, resolution, *, skimage_method="li", mark_dim_centers=False):
    """
    Segment an optical image of the solar photosphere into tri-value maps with:

     * 0 as intergranule
     * 0.5 as faculae ->  NO, 1.5, RIGHT??
     * 1 as granule

    Parameters
    ----------
    smap : `~sunpy.map.GenericMap`
        `~sunpy.map.GenericMap` containing data to segment.
    resolution : `float`
        Spatial resolution (arcsec/pixel) of the data.
    skimage_method : {"li", "otsu", "isodata", "mean", "minimum", "yen", "triangle"}
        scikit-image thresholding method, defaults to "li".
        Depending on input data, one or more of these methods may be
        significantly better or worse than the others. Typically, 'li', 'otsu',
        'mean', and 'isodata' are good choices, 'yen' and 'triangle' over-
        identify intergranule material, and 'minimum' over identifies granules.
    mark_dim_centers : `bool`
        Whether to mark dim granule centers as a separate category for future exploration.

    Returns
    -------
    segmented_map : `~sunpy.map.GenericMap`
        `~sunpy.map.GenericMap` containing a segmented image (with the original header).
    """
    if not isinstance(smap, sunpy.map.mapbase.GenericMap):
        raise TypeError("Input must be an instance of a sunpy.map.GenericMap")
    if skimage_method not in METHODS:
        raise TypeError("Method must be one of: " + ", ".join(METHODS))

    median_filtered = sndi.median_filter(smap.data, size=3)
    print('median filtered')
    # Apply initial skimage threshold.
    threshold = get_threshold(median_filtered, skimage_method)
    segmented_image = np.uint8(median_filtered > threshold)
    print('applied threshold')
    # Fix the extra intergranule material bits in the middle of granules.
    seg_im_fixed = trim_intergranules(segmented_image, mark=mark_dim_centers)
    # Mark faculae and get final granule and facule count.
    seg_im_markfac, faculae_count, granule_count = mark_faculae(seg_im_fixed, smap.data, resolution)
    logging.info(f"Segmentation has identified {granule_count} granules and {faculae_count} faculae")
    segmented_map = sunpy.map.Map(seg_im_markfac, smap.meta)
    return segmented_map


def get_threshold(data, method):
    """
    Get the threshold value using given skimage segmentation type.

    Parameters
    ----------
    data : `numpy.ndarray`
        Data to threshold.
    method : {"li", "otsu", "isodata", "mean", "minimum", "yen", "triangle"}
        scikit-image thresholding method.

    Returns
    -------
    threshold : `float`
        Threshold value.
    """

    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be an instance of a np.ndarray")
    elif method == "li":
        threshold = skimage.filters.threshold_li(data)
    if method == "otsu":
        threshold = skimage.filters.threshold_otsu(data)
    elif method == "yen":
        threshold = skimage.filters.threshold_yen(data)
    elif method == "mean":
        threshold = skimage.filters.threshold_mean(data)
    elif method == "minimum":
        threshold = skimage.filters.threshold_minimum(data)
    elif method == "triangle":
        threshold = skimage.filters.threshold_triangle(data)
    elif method == "isodata":
        threshold = skimage.filters.threshold_isodata(data)
    # else:
    #     raise ValueError("Method must be one of: " + ", ".join(METHODS))
    return threshold


def trim_intergranules(segmented_image, mark=False):
    """
    Remove the erroneous identification of intergranule material in the middle
    of granules that the pure threshold segmentation produces.

    Parameters
    ----------
    segmented_image : `numpy.ndarray`
        The segmented image containing incorrect extra intergranules.
    mark : `bool`
        If `False` (the default), remove erroneous intergranules.
        If `True`, mark them as 0.5 instead (for later examination).

    Returns
    -------
    segmented_image_fixed : `numpy.ndarray`
        The segmented image without incorrect extra intergranules.
    """

    if len(np.unique(segmented_image)) > 2:
        raise ValueError("segmented_image must only have values of 1 and 0.")
    segmented_image_fixed = np.copy(segmented_image).astype(float)  # Float conversion for correct region labeling.
    labeled_seg = skimage.measure.label(segmented_image_fixed + 1, connectivity=2)
    values = np.unique(labeled_seg)
    # Find value of the large continuous 0-valued region.
    size = 0
    print(f'\tloop 1 to {len(values)} (should take like 2 minutes)')
    for value in values:
        if len((labeled_seg[labeled_seg == value])) > size: # if bigger than previous largest
            if sum(segmented_image[labeled_seg == value] == 0): # if a zero (IG) region
                real_IG_value = value
                size = len(labeled_seg[labeled_seg == value])
    # Set all other 0 regions to mark value (1 or 0.5).
    print(f'\tloop 2 to {len(values)} (should take like 2 minutes)')
    for value in values:
        if np.sum(segmented_image[labeled_seg == value]) == 0:
            if value != real_IG_value:
                if not mark:
                    segmented_image_fixed[labeled_seg == value] = 1
                elif mark:
                    segmented_image_fixed[labeled_seg == value] = 0.5
    return segmented_image_fixed


def mark_faculae(segmented_image, data, resolution):
    """
    Mark faculae separately from granules - give them a value of 1.5 not 1.

    Parameters
    ----------
    segmented_image : `numpy.ndarray`
        The segmented image containing incorrect middles.
    data : `numpy array`
        The original image.
    resolution : `float`
        Spatial resolution (arcsec/pixel) of the data.

    Returns
    -------
    segmented_image_fixed : `numpy.ndrray`
        The segmented image with faculae marked as 1.5.
    faculae_count: `int`
        The number of faculae identified in the image.
    granule_count: `int`
        The number of granules identified, after re-classifcation of faculae.
    """
    fac_size_limit = 2  # Max size of a faculae in square arcsec.
    fac_pix_limit = fac_size_limit / resolution # SHOULD SQUARE THIS???? 
    fac_brightness_limit = np.nanmean(data) + 0.5 * np.nanstd(data) # General flux limit determined by visual inspection.
    if len(np.unique(segmented_image)) > 3:
        raise ValueError("segmented_image must have only values of 1, 0 and a 0.5 (if dim centers marked)")
    segmented_image_fixed = np.copy(segmented_image.astype(float))
    labeled_seg = skimage.measure.label(segmented_image + 1, connectivity=2)
    values = np.unique(labeled_seg)
    fac_count = 0
    print(f'\tloop 3 to {len(values)} (this is the sticking point)')
    for value in values:
        print(f'\t\t{value}', end='\r')
        mask = np.zeros_like(segmented_image)
        mask[labeled_seg == value] = 1
        # Check that is a 1 (white) region.
        if np.sum(np.multiply(mask, segmented_image)) > 0:
            region_size = len(segmented_image_fixed[mask == 1])
            tot_flux = np.sum(data[mask == 1])
            # check that region is small.
            if region_size < fac_pix_limit:
                # Check that avg flux very high.
                if tot_flux / region_size > fac_brightness_limit:
                    segmented_image_fixed[mask == 1] = 1.5
                    fac_count += 1
    gran_count = len(values) - 1 - fac_count  # Subtract 1 for IG region.
    return segmented_image_fixed, fac_count, gran_count

def convert_back(seg, to='binary'):
    '''
    Convert 4-value seg into binary or tri-value (so that dont have to rerun seg algorithm)
    Both DM *and* BPs (if to=binary) are converted to GR
    '''
    outseg = np.copy(seg)
    outseg[seg == 0.5] = 1
    if to=='binary': outseg[seg == 1.5] = 1

    return outseg

    # #### PLACE THIS CODE WITHIN TRIM_INTERGRGRANULES FUNCTION ###
    # print(real_IG_value)
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(50, 10))
    # im = ax1.imshow(segmented_image[0:2000, 0:2000], origin='lower')
    # plt.colorbar(im, ax=ax1)
    # im = ax2.imshow(labeled_seg[0:2000, 0:2000], origin='lower')
    # plt.colorbar(im, ax=ax2); ax2.set_title('label values')
    # region1, region2, region3 = np.zeros_like(labeled_seg)*np.NaN, np.zeros_like(labeled_seg)*np.NaN, np.zeros_like(labeled_seg)*np.NaN
    # region1[labeled_seg==values[0]] = values[0]; print(np.where(labeled_seg==values[0]))
    # region2[labeled_seg==values[1]] = values[1]
    # region3[labeled_seg==values[2]] = values[2]; region3[labeled_seg==values[1]] = values[1] 
    # im = ax3.imshow(region1[0:2000, 0:2000], origin='lower')
    # plt.colorbar(im, ax=ax3); ax3.set_title('value 1')
    # im = ax4.imshow(region2[0:2000, 0:2000], origin='lower')
    # plt.colorbar(im, ax=ax4); ax4.set_title('value 2')
    # im = ax5.imshow(region3[0:2000, 0:2000], origin='lower')
    # plt.colorbar(im, ax=ax5); ax5.set_title('value 3 and 2')
    # plt.savefig('test0622')
    # a=b
    ###################

######## UPDATES OF SUNKIT-IMAGE VERSIONS OF SEG FUNCTIONS, USING NEW (V2) METHODS ######

import skimage
import matplotlib
import scipy
import logging

def segment_2(smap, *, skimage_method="li", mark_dim_centers=False, bp_min_flux=None):
    """
    Segment an optical image of the solar photosphere into tri-value maps with:

     * 0 as intergranule
     * 1 as granule
     * 2 as brightpoint

    If mark_dim_centers is set to True, an additional label, 3, will be assigned to
    dim grnanule centers.

    Parameters
    ----------
    smap : `~sunpy.map.GenericMap`
        `~sunpy.map.GenericMap` containing data to segment. Must have square pixels.
    skimage_method : {"li", "otsu", "isodata", "mean", "minimum", "yen", "triangle"}, optional
        scikit-image thresholding method, defaults to "li".
        Depending on input data, one or more of these methods may be
        significantly better or worse than the others. Typically, 'li', 'otsu',
        'mean', and 'isodata' are good choices, 'yen' and 'triangle' over-
        identify intergranule material, and 'minimum' over identifies granules.
    mark_dim_centers : `bool`, optional
        Whether to mark dim granule centers as a separate category for future exploration.
    bp_min_flux : `float`, optional
        Minimum flux per pixel for a region to be considered a brightpoint.
        Default is `None` which will use data mean + 0.5 * sigma.

    Returns
    -------
    segmented_map : `~sunpy.map.GenericMap`
        `~sunpy.map.GenericMap` containing a segmented image (with the original header).
    """
    if not isinstance(smap, sunpy.map.mapbase.GenericMap):
        raise TypeError("Input must be an instance of a sunpy.map.GenericMap")
    if smap.scale[0].value == smap.scale[1].value:
        resolution = smap.scale[0].value
    else:
        raise ValueError("Currently only maps with square pixels are supported.")
    # Obtain local histogram equalization of map.
    if not isinstance(smap, sunpy.map.mapbase.GenericMap):
        raise TypeError("Input must be an instance of a sunpy.map.GenericMap")
    if smap.scale[0].value == smap.scale[1].value:
        resolution = smap.scale[0].value
    else:
        raise ValueError("Currently only maps with square pixels are supported.")
    # Obtain local histogram equalization of map.
    map_norm = ((smap.data - np.nanmin(smap.data))/(np.nanmax(smap.data) - np.nanmin(smap.data))) # min-max normalization to [0, 1] 
    map_he = skimage.filters.rank.equalize(skimage.util.img_as_ubyte(map_norm), footprint=skimage.morphology.disk(radius=100))
    # Apply initial skimage threshold.
    median_filtered = scipy.ndimage.median_filter(map_he, size=3)
    threshold = _get_threshold_2(median_filtered, skimage_method)
    segmented_image = np.uint8(median_filtered > threshold)
    # Fix the extra intergranule material bits in the middle of granules.
    seg_im_fixed = _trim_intergranules_2(segmented_image, mark=mark_dim_centers)
    # Mark brightpoint and get final granule and brightpoint count.
    seg_im_markbp, brightpoint_count, granule_count = _mark_brightpoint_2(
        seg_im_fixed, smap.data, map_he, resolution, bp_min_flux
    )
    logging.info(f"Segmentation has identified {granule_count} granules and {brightpoint_count} brightpoint")
    # Create output map using input wcs and adding colormap such that 0 (intergranules) = black, 1 (granule) = white, 2 (brightpoints) = yellow, 3 (dim_centers) = blue.
    segmented_map = sunpy.map.Map(seg_im_markbp, smap.wcs)
    cmap = matplotlib.colors.ListedColormap(["black", "white", "#ffc406", "blue"])
    norm = matplotlib.colors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=cmap.N)
    segmented_map.plot_settings["cmap"] = cmap
    segmented_map.plot_settings["norm"] = norm

    segmented_map = seg_im_markbp

    return segmented_map

def _get_threshold_2(data, method):
    """
    Get the threshold value using given skimage segmentation type.

    Parameters
    ----------
    data : `numpy.ndarray`
        Data to threshold.
    method : {"li", "otsu", "isodata", "mean", "minimum", "yen", "triangle"}
        scikit-image thresholding method.

    Returns
    -------
    threshold : `float`
        Threshold value.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be an instance of a np.ndarray")
    if len(data.flatten()) > 500**2:
        data = np.random.choice(data.flatten(), (500, 500)) # Computing threshold based on random sample works well and saves significant computatonal time
    method = method.lower()
    method_funcs = {
        "li": skimage.filters.threshold_li,
        "otsu": skimage.filters.threshold_otsu,
        "yen": skimage.filters.threshold_yen,
        "mean": skimage.filters.threshold_mean,
        "minimum": skimage.filters.threshold_minimum,
        "triangle": skimage.filters.threshold_triangle,
        "isodata": skimage.filters.threshold_isodata,
    }
    if method not in method_funcs:
        raise ValueError("Method must be one of: " + ", ".join(list(method_funcs.keys())))
    threshold = method_funcs[method](data)
    return threshold


def _trim_intergranules_2(segmented_image, mark=False):
    """
    Remove the erroneous identification of intergranule material in the middle
    of granules that the pure threshold segmentation produces.

    Parameters
    ----------
    segmented_image : `numpy.ndarray`
        The segmented image containing incorrect extra intergranules.
    mark : `bool`
        If `False` (the default), remove erroneous intergranules.
        If `True`, mark them as 3 instead (for later examination).

    Returns
    -------
    segmented_image_fixed : `numpy.ndarray`
        The segmented image without incorrect extra intergranules.
    """
    if len(np.unique(segmented_image)) > 2:
        raise ValueError("segmented_image must only have values of 1 and 0.")
    # Float conversion for correct region labeling.
    segmented_image_fixed = np.copy(segmented_image).astype(float)
    # Add padding of intergranule around edges. Aviods the case where all edge pixels are granule, which will result in all dim centers as intergranules.
    pad = int(np.shape(segmented_image)[0]/200)
    segmented_image_fixed[:,0:pad] = 0 
    segmented_image_fixed[0:pad,:] = 0 
    segmented_image_fixed[:,-pad:] = 0 
    segmented_image_fixed[-pad:,:] = 0 
    labeled_seg = skimage.measure.label(segmented_image_fixed + 1, connectivity=2)
    values = np.unique(labeled_seg)
    # Find value of the large continuous 0-valued region.
    size = 0
    for value in values:
        if len((labeled_seg[labeled_seg == value])) > size and sum(segmented_image[labeled_seg == value] == 0):
            real_IG_value = value
            size = len(labeled_seg[labeled_seg == value])
    # Set all other 0 regions to mark value (3).
    for value in values:
        if np.sum(segmented_image[labeled_seg == value]) == 0:
            if value != real_IG_value:
                if not mark:
                    segmented_image_fixed[labeled_seg == value] = 1
                elif mark:
                    segmented_image_fixed[labeled_seg == value] = 3
    return segmented_image_fixed


def _mark_brightpoint_2(segmented_image, data, HE_data, resolution, bp_min_flux=None):
    """
    Mark brightpoints separately from granules - give them a value of 2.

    Parameters
    ----------
    segmented_image : `numpy.ndarray`
        The segmented image containing incorrect middles.
    data : `numpy array`
        The original image.
    HE_data : `numpy array`
        Original image with local histogram equalization applied.
    resolution : `float`
        Spatial resolution (arcsec/pixel) of the data.
    bp_min_flux : `float`, optional
        Minimum flux per pixel for a region to be considered a brightpoint.
        Default is `None` which will use data mean + 0.5 * sigma.

    Returns
    -------
    segmented_image_fixed : `numpy.ndrray`
        The segmented image with brightpoints marked as 2.
    brightpoint_count: `int`
        The number of brightpoints identified in the image.
    granule_count: `int`
        The number of granules identified, after re-classifcation of brightpoint.
    """
    # General size limits 
    bp_size_limit = (
        0.1  # Approximate max size of a photosphere bright point in square arcsec (see doi 10.3847/1538-4357/aab150)
    )
    bp_pix_upper_limit = (bp_size_limit / resolution)**2 # Max area in pixels
    bp_pix_lower_limit = 4  # Very small bright regions are likely artifacts
    # General flux limit determined by visual inspection (set using equalized map)
    if bp_min_flux is None:
        stand_devs = 1.25
        bp_brightness_limit = np.nanmean(HE_data) + stand_devs*np.nanstd(HE_data)
    else:
        bp_brightness_limit = bp_min_flux
    if len(np.unique(segmented_image)) > 3:
        raise ValueError("segmented_image must have only values of 1, 0 and 3 (if dim centers marked)")
    # Obtain gradient map and set threshold for gradient on BP edges
    grad = np.abs(np.gradient(data)[0] + np.gradient(data)[1])
    bp_min_grad = np.quantile(grad, 0.95)
    # Label all regions of flux greater than brightness limit (candidate regions)
    bright_dim_seg = np.zeros_like(data)
    bright_dim_seg[HE_data > bp_brightness_limit] = 1
    labeled_bright_dim_seg = skimage.measure.label(bright_dim_seg + 1, connectivity=2)
    values = np.unique(labeled_bright_dim_seg)
    # From candidate regions, select those within pixel limit and gradient limit
    segmented_image_fixed = np.copy(segmented_image.astype(float))  # Make type float to enable adding float values
    bp_count = 0
    for value in values:
        if (bright_dim_seg[labeled_bright_dim_seg==value])[0]==1: # Check region is not the non-bp region
            # check that region is within pixel limits.
            region_size = len(labeled_bright_dim_seg[labeled_bright_dim_seg==value])
            if region_size < bp_pix_upper_limit and region_size > bp_pix_lower_limit:
                # check that region has high average gradient (maybe try max gradient?)
                region_mean_grad = np.mean(grad[labeled_bright_dim_seg==value])
                if region_mean_grad > bp_min_grad:
                    segmented_image_fixed[labeled_bright_dim_seg==value] = 2
                    bp_count += 1
    gran_count = len(values) - 1 - bp_count  # Subtract 1 for IG region.
    return segmented_image_fixed, bp_count, gran_count


def segments_overlap_fraction(segment1, segment2):
    """
    Compute the fraction of overlap between two segmented SunPy Maps.

        Designed for comparing output Map from `segment` with other segmentation methods.

    Parameters
    ----------
    segment1: `~sunpy.map.GenericMap`
        Main `~sunpy.map.GenericMap` to compare against. Must have 0 = intergranule, 1 = granule.
    segment2 :`~sunpy.map.GenericMap`
        Comparison `~sunpy.map.GenericMap`. Must have 0 = intergranule, 1 = granule.
        As an example, this could come from a simple segment using sklearn.cluster.KMeans

    Returns
    -------
    confidence : `float`
        The numeric confidence metric: 0 = no agreement and 1 = complete agreement.
    """
    segment1 = np.array(segment1.data)
    segment2 = np.array(segment2.data)
    total_granules = np.count_nonzero(segment1 == 1)
    total_intergranules = np.count_nonzero(segment1 == 0)
    if total_granules == 0:
        raise ValueError("No granules in `segment1`. It is possible the clustering failed.")
    if total_intergranules == 0:
        raise ValueError("No intergranules in `segment1`. It is possible the clustering failed.")
    granule_agreement_count = 0
    intergranule_agreement_count = 0
    granule_agreement_count = ((segment1 == 1) * (segment2 == 1)).sum()
    intergranule_agreement_count = ((segment1 == 0) * (segment2 == 0)).sum()
    percentage_agreement_granules = granule_agreement_count / total_granules
    percentage_agreement_intergranules = intergranule_agreement_count / total_intergranules
    confidence = np.mean([percentage_agreement_granules, percentage_agreement_intergranules])
    return confidence

#### other ####
def fits_to_map(filename):
    """
    Read .fits file data into a sunpy map.
    ----------
    Parameters:
        filename (string): Path to input data file (.fits format)
    ----------
    Returns:
        data_map: SunPy map containing the data and header
    """

    try:
        hdu = fits.open(filename)
        data = hdu[0].data
    except FileNotFoundError:
        raise FileNotFoundError('Cannot find ' + filename)
    except Exception:
        raise Exception('Data does not appear to be in correct .fits format')

    data_map = sunpy.map.Map(filename)

    return data_map
