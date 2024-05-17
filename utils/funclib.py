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
from tqdm import tqdm

######## Functions for NNs

# Dataset
class MyDataset(Dataset):

    def __init__(self, image_dir, mask_dir, set, norm=False, channels=['X'], n_classes=2, randomSharp=False, im_size=None, no_first=False):
        self.image_dir = f'{image_dir}{set}'
        self.mask_dir = f'{mask_dir}{set}'
        self.set = set
        self.images = os.listdir(f'{image_dir}{set}')
        self.norm = norm
        self.channels = channels
        self.n_classes = n_classes
        self.randomSharp = randomSharp
        self.im_size = im_size
        self.transform = transforms.Resize(im_size)
        self.no_first = no_first

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Get image
        img_path = os.path.join(self.image_dir, self.images[index]) # path to one data image (or SET of [20, npix, npix] if timeseries)
        img = np.load(img_path)
        if np.max(img > 1):
            raise ValueError('This image does not appear to come from a pre-normalized set')
        if img.dtype.byteorder == '>':
            img = img.newbyteorder().byteswap() 
        # if self.im_size != None: # cut to desired size, e.g. to make divisible by 2 5 times, for WNet
        #     img = np.array(self.transform(torch.from_numpy(np.expand_dims(img, axis=0)))).squeeze()
        if self.randomSharp: # add 50% chance of image being blurred/sharpened by a factor pulled from a skewed guassian (equal chance of 1/4 and 4)
            img =((img - np.nanmin(img))/(np.nanmax(img) - np.nanmin(img))) # first must [0, 1] normalize
            img = torch.from_numpy(np.expand_dims(img, axis=0)) # transforms expect a batch dimension
            n = stats.halfnorm.rvs(loc=1, scale=1, size=1)[0]
            s = n if np.random.rand(1)[0] < 0.5 else 1/n
            transf = transforms.RandomAdjustSharpness(sharpness_factor=s, p=0.5)
            img = transf(img)[0] # remove batch dimension for now
        # if self.norm:  # normalize DONT DO THIS!!!!! INSTEAD USE NPY SECTIONS CREATED FROM NORMED OG FILES!!!!
        #     img = (img - np.mean(img))/np.std(img) # normalize to std normal dist
        if self.im_size != None: # cut to desired size, e.g. to make divisible by 2 5 times, for WNet
            img = np.array(self.transform(torch.from_numpy(np.expand_dims(img, axis=0)))).squeeze()
        if self.channels != ['X']: # Add feature layers
            if self.channels[0].startswith('timeseries'):
                tag = self.channels[0][self.channels[0].find('ies')+3:]
                image = fill_timeseries(img, tag)
            else:
                image = np.zeros((len(self.channels), img.shape[0], img.shape[1]), dtype=np.float32) # needs to be float32 not float64
                image[0, :, :] = img
                for i in range(1, len(self.channels)):
                    image[i, :, :] = get_feature(img, self.channels[i], index, self.images[index], self.set, img_path, self.transform)
        else: # Add dummy axis
            image = np.zeros((1, img.shape[0], img.shape[1]), dtype=np.float32) # needs to be float32 not float64
            image[0, :, :] = img
        if self.no_first:
            image = image[1:, :, :]
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
            #mask = mask[:, 0:self.im_size, 0:self.im_size]
            mask = np.array(self.transform(torch.from_numpy(mask)))

        return image, mask

def fill_timeseries(img, tag):
    if tag == '20_5':
        image = np.zeros((5, img.shape[1], img.shape[2]), dtype=np.float32) # needs to be float32 not float64
        image[0, :, :] = img[0, :, :]
        image[1, :, :] = img[5, :, :]
        image[2, :, :] = img[10, :, :] # target image (is it important to put this here?)
        image[3, :, :] = img[15, :, :]
        image[4, :, :] = img[19, :, :] # should've probabaly done sets of 21.. oh well
    if tag == '40_5':
        image = np.zeros((5, img.shape[1], img.shape[2]), dtype=np.float32) # needs to be float32 not float64
        image[0, :, :] = img[0, :, :]
        image[1, :, :] = img[10, :, :]
        image[2, :, :] = img[20, :, :] # target image (is it important to put this here?)
        image[3, :, :] = img[30, :, :]
        image[4, :, :] = img[40, :, :] # these sets are 41
    if tag == '40_9':
        image = np.zeros((9, img.shape[1], img.shape[2]), dtype=np.float32) # needs to be float32 not float64
        image[0, :, :] = img[0, :, :]
        image[1, :, :] = img[5, :, :]
        image[2, :, :] = img[10, :, :] 
        image[3, :, :] = img[15, :, :]
        image[4, :, :] = img[20, :, :] # target image (is it important to put this here?)
        image[5, :, :] = img[25, :, :]
        image[6, :, :] = img[30, :, :]
        image[7, :, :] = img[35, :, :] 
        image[8, :, :] = img[40, :, :] # these sets are 41
    return image

def get_feature(img, name, index, image_name, set, imgpath, transform):
    if name == 'gradx': a = np.gradient(img)[0]
    elif name == 'grady': a = np.gradient(img)[1]
    elif name == 'smoothed': a = scipy.ndimage.gaussian_filter(img, sigma=3)
    elif 'power' in name:
        n = int(name[-1])
        a = img**n
    # elif name == 'deltaBinImg':
    #     UNet1seg = np.load(f'../UNet1_outputs/pred_{index}.npy') # zeros and ones
    #     imgnorm = (img - np.mean(img))/np.std(img) # normalize to range [0, 1]
    #     a = UNet1seg - imgnorm # difference between binary segmentation and image (pos for bp, neg for dm)
    elif name == 'binary_residual':
        im_scaled = (img-np.min(img))/(np.max(img)-np.min(img)) # (im-np.nanmean(im))/np.nanstd(im)
        if set == 'train':
            if os.path.exists(f'../NN_outputs/WNet8m_outputs/predict_on_train/wnet8seg_{index}'): # if i've already saved them
                wnet8_preds = np.squeeze(np.load(f'../NN_outputs/WNet8m_outputs/predict_on_train/wnet8seg_{index}'))
            else: 
                model = MyWNet(squeeze=2, ch_mul=64, in_chans=2, out_chans=2)
                model.load_state_dict(torch.load(f'../NN_storage/WNet8m.pth'))
                x = np.zeros((1, 2, img.shape[0], img.shape[1]), dtype=np.float32) # needs to be float32 not float64
                x[0, 0, :, :] = img
                x[0, 1, :, :] = img**2
                X = torch.from_numpy(x) # transforms.Resize(128)(torch.from_numpy(x))
                probs = model(X, returns='enc') # defualt is to return dec, but we want seg
                wnet8_preds = np.argmax(probs.detach().numpy(), axis=1).astype(float).squeeze() # turn probabilities [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
                np.save(f'../NN_outputs/WNet8m_outputs/predict_on_train/wnet8seg_train/wnet8seg_{index}', wnet8_preds)
        if set == 'val':
            wnet8_preds = np.squeeze(np.load(f'../NN_outputs/WNet8m_outputs/pred_{index}.npy'))
        kernel = np.ones((30,30))/900
        wnet8_preds_smooth = cv2.filter2D(wnet8_preds, -1, kernel)
        a = (wnet8_preds_smooth - im_scaled)**2 
    elif name == 'Bz':
        mag_path = f'../Data/UNetData_MURaM/mag_images/{set}/{image_name}' # path to one mag image THIS SHOULD ENSURE ITS THE MAG CORRESPONDING TO THE CORRECT IMAGE
        mag = np.load(mag_path).newbyteorder().byteswap()
        #a = (mag - np.mean(mag))/np.std(mag) # normalize to std normal dist 
        mag = mag**2
        a = (mag - np.mean(mag))/np.std(mag) # normalize to std normal dist
    elif name == 'median_residual':
        meddir = f'{imgpath[0:imgpath.find("norm_images/")]}med8_images/{set}/'
        if os.path.exists(meddir):
            #imgname = imgpath[imgpath.find("VBI"):]
            med_path = f'{meddir}med8_{image_name}'
            med = np.load(med_path)
            if np.shape(med) != np.shape(img): # already cut down image
                med = np.array(transform(torch.from_numpy(np.expand_dims(med, axis=0)))).squeeze()
            a = img - med
        else: 
            if set == 'val':
                s = 8
                a = img - sndi.median_filter(img, size=s)
            else:
                raise FileNotFoundError(f'Median filtered images not saved for this set ({meddir} does not exist).')
        if 'MURAM' in imgpath: 
            mean = 0.000368; sd = 0.019967
            a = (a - mean)/sd
    else: raise ValueError(f'Channel name {name} not recognized')

    return a

# Containor for multiple layers (for convenience)
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        # Call init method inherated from nn.Module
        super(DoubleConv, self).__init__()
        print('')
        # Define layer that is a bunch of layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x): # this is what gets called when you call DoubleConv

        # print(f'\t\tCalling MyUNet.DoubleConv foward with x shape {x.shape}, with first Conv2d layer {self.conv[0]}')
        return self.conv(x)


# UNet
class MyUNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512],):

        #print('Initializing MyUNet')
        
        # Initialize
        super(MyUNet, self).__init__() # Call init method inherated from nn.Module
        # Set empty layer type lists
        self.ups = nn.ModuleList() 
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Define pooling layer (recall: this is not saying the pooling layer is called after the up and down stuff; its just defining the up, down, and pool attributes all together here)
        # Fill list of down layers 
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature  # after each convolution we set (next) in_channel to (previous) out_channels  
        # Fill list of up layers 
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2,))
            self.ups.append(DoubleConv(feature*2, feature))
        # Define layer at the bottom
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # Define last conv to get correct output num channels
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        #print('Finished calling _init_')
    

    def forward(self, x): # this is what gets called when you call MyUNet

        # Initialize list to store skip connections
        skip_connections = []
        # Do all the downs in self.downs
        for down in self.downs:
            #print(f'\tperforming down {down} ')
            x = down(x) # the down layers are all MyUNet.DoubleConv, which is a set of layers within a nn.Sequential containor 
            skip_connections.append(x) # NEED TO LEARN ABOUT SKIP_CONNECTIONS
            x = self.pool(x)
        # Perform bottleneck layer
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # reverse 
        # Do all the ups in self.ups
        for idx in range(0, len(self.ups), 2): # step of 2 becasue add conv step
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=None)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


# Function to train one epoch
def train_UNET(loader, model, optimizer, loss_fn, scaler, dm_weight=1, bp_weight=1, device='cpu'):
    
    loop = tqdm(loader) # is this a progress bar? but then below its looping through loop

    # Train on each set in train loader
    for batch_idx, (data, targets) in enumerate(loop):
        # set to use device
        data = data.to(device)
        targets = targets.float().to(device)

        # forward
        predictions = model(data) # call model to get predictions (not class probs; contains negs; apply sigmoid to get probs).
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
        loop.set_postfix(loss=loss.item()) #; print(f'\t\tBatch {batch_idx}, {loss_fn}: {loss}')

# Function to calculate validation accuracy after one training epoch
def validate(val_loader, model, device='cpu'):

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval() # set model into eval mode
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)#.unsqueeze(1)
            preds = torch.sigmoid(model(x)) # call model to get predictions
            if preds.shape[1] == 1: # if binary (predictions have 1 layer)
                preds = (preds > 0.5).float() # turn regression value (between zero and 1, e.g. "prob of being 1") into predicted 0s and 1s
            else: # if muliclasss (predictions have n_classes layers)
                preds = np.argmax(preds.detach().numpy(), axis=1) # turn probabilities [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
                y = np.argmax(y.detach().numpy(), axis=1) # turn 1-hot-encoded layers [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
            num_correct += len(np.where(preds == y)[0]) #(preds == y).sum()
            num_pixels += len(preds.flatten()) # torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
    accuracy = num_correct/num_pixels*100

    return accuracy, dice_score

# Function to save results of trained model on validation data
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
            preds = preds.detach().numpy()[:,0,:,:] # NOTE: currently I am still constructing binary inputs/truths as 2 layers where second is just dummy layer
            y = y.detach().numpy()[:,0,:,:] # NOTE: currently I am still constructing binary inputs/truths as 2 layers where second is just dummy layer
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


# Container for multiple layers
class Block(nn.Module):
    def __init__(self, in_filters, out_filters, seperable=True, padding_mode=None):
        super(Block, self).__init__()
        
        if seperable:
            self.spatial1=nn.Conv2d(in_filters, in_filters, kernel_size=3, groups=in_filters, padding=1, padding_mode=padding_mode)
            self.depth1=nn.Conv2d(in_filters, out_filters, kernel_size=1, padding_mode=padding_mode)
            self.conv1=lambda x: self.depth1(self.spatial1(x))
            self.spatial2=nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, groups=out_filters, padding_mode=padding_mode)
            self.depth2=nn.Conv2d(out_filters, out_filters, kernel_size=1, padding_mode=padding_mode)
            self.conv2=lambda x: self.depth2(self.spatial2(x))
            
        else:
            self.conv1=nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1, padding_mode=padding_mode)
            self.conv2=nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, padding_mode=padding_mode)

        # self.relu1 = nn.ReLU(); self.dropout1 = nn.Dropout(0.65)  # from reproduction
        self.batchnorm1=nn.BatchNorm2d(out_filters)
        # self.relu2 = nn.ReLU(); self.dropout2 = nn.Dropout(0.65)  # from reproduction
        self.batchnorm2=nn.BatchNorm2d(out_filters) 

    def forward(self, x):

        x=self.batchnorm1(self.conv1(x)).clamp(0) 
        # x = self.relu1(x); x = self.dropout1(x)  # from reproduction
        x=self.batchnorm2(self.conv2(x)).clamp(0)
        # x = self.relu2(x); x = self.dropout2(x)  # from reproduction

        return x

# Encoder UNet
class UEnc(nn.Module):
    def __init__(self, squeeze, ch_mul=64, in_chans=3, padding_mode=None):
        super(UEnc, self).__init__()
        
        self.enc1=Block(in_chans, ch_mul, seperable=False, padding_mode=padding_mode)
        self.enc2=Block(ch_mul, 2*ch_mul, padding_mode=padding_mode)
        self.enc3=Block(2*ch_mul, 4*ch_mul, padding_mode=padding_mode)
        self.enc4=Block(4*ch_mul, 8*ch_mul, padding_mode=padding_mode)
        
        self.middle=Block(8*ch_mul, 16*ch_mul, padding_mode=padding_mode)
        
        self.up1=nn.ConvTranspose2d(16*ch_mul, 8*ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1) # padding_mode only allowed to be 'zeros'
        self.dec1=Block(16*ch_mul, 8*ch_mul, padding_mode=padding_mode)
        self.up2=nn.ConvTranspose2d(8*ch_mul, 4*ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2=Block(8*ch_mul, 4*ch_mul, padding_mode=padding_mode)
        self.up3=nn.ConvTranspose2d(4*ch_mul, 2*ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3=Block(4*ch_mul, 2*ch_mul, padding_mode=padding_mode)
        self.up4=nn.ConvTranspose2d(2*ch_mul, ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec4=Block(2*ch_mul, ch_mul, seperable=False, padding_mode=padding_mode)
        
        self.final=nn.Conv2d(ch_mul, squeeze, kernel_size=(1, 1), padding_mode=padding_mode) # self.final=nn.Conv2d(ch_mul, squeeze, kernel_size=(1, 1)); self.softmax = nn.Softmax2d()
        
    def forward(self, x): 
        
        enc1=self.enc1(x) # [16, 1, 140, 140] -> [16, 64, 140, 140]  
        enc2=self.enc2(F.max_pool2d(enc1, (2,2))) # [16, 64, 140, 140] -> [16, 128, 70, 70]         
        enc3=self.enc3(F.max_pool2d(enc2, (2,2))) # [16, 128, 70, 70] -> [16, 256, 35, 35]         
        enc4=self.enc4(F.max_pool2d(enc3, (2,2))) # [16, 256, 35, 35] -> [16, 512, 17, 17] 
        
        middle=self.middle(F.max_pool2d(enc4, (2,2))) # [16, 512, 17, 17] -> [16, 1024, 8, 8]
        
        up1=torch.cat([enc4, self.up1(middle)], 1) # [16, 512, 17, 17] + self.up1(middle):[16, 512, 16, 16] 
        dec1=self.dec1(up1)
        up2=torch.cat([enc3, self.up2(dec1)], 1)
        dec2=self.dec2(up2)
        up3=torch.cat([enc2, self.up3(dec2)], 1)
        dec3=self.dec3(up3)
        up4=torch.cat([enc1, self.up4(dec3)], 1)
        dec4=self.dec4(up4)
        
        final=self.final(dec4)

        # print(f'x {x.shape}')
        # print(f'enc1 {enc1.shape}')
        # print(f'enc2 {enc2.shape}')
        # print(f'enc3 {enc3.shape}')
        # print(f'enc4 {enc4.shape}')
        # print(f'middle {middle.shape}')
        # print(f'up1 {up1.shape}')
        # print(f'dec1 {dec1.shape}')
        # print(f'up2 {up2.shape}')
        # print(f'dec2 {dec2.shape}')
        # print(f'up3 {up3.shape}')
        # print(f'dec3 {dec3.shape}')
        # print(f'up4 {up4.shape}')
        # print(f'dec4 {dec4.shape}')
        # print(f'final {final.shape}')
        
        return final

# Decoder UNet
class UDec(nn.Module):
    def __init__(self, squeeze, ch_mul=64, in_chans=3, padding_mode=None):
        super(UDec, self).__init__()
        
        self.enc1=Block(squeeze, ch_mul, seperable=False, padding_mode=padding_mode)
        self.enc2=Block(ch_mul, 2*ch_mul, padding_mode=padding_mode)
        self.enc3=Block(2*ch_mul, 4*ch_mul, padding_mode=padding_mode)
        self.enc4=Block(4*ch_mul, 8*ch_mul, padding_mode=padding_mode)
        
        self.middle=Block(8*ch_mul, 16*ch_mul, padding_mode=padding_mode)
        
        self.up1=nn.ConvTranspose2d(16*ch_mul, 8*ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1) # padding_mode only allowed to be 'zeros'
        self.dec1=Block(16*ch_mul, 8*ch_mul, padding_mode=padding_mode)
        self.up2=nn.ConvTranspose2d(8*ch_mul, 4*ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2=Block(8*ch_mul, 4*ch_mul, padding_mode=padding_mode)
        self.up3=nn.ConvTranspose2d(4*ch_mul, 2*ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3=Block(4*ch_mul, 2*ch_mul, padding_mode=padding_mode)
        self.up4=nn.ConvTranspose2d(2*ch_mul, ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec4=Block(2*ch_mul, ch_mul, seperable=False, padding_mode=padding_mode)
        
        self.final=nn.Conv2d(ch_mul, in_chans, kernel_size=(1, 1), padding_mode=padding_mode)
        
    def forward(self, x):
        
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, (2,2)))
        enc3 = self.enc3(F.max_pool2d(enc2, (2,2)))
        enc4 = self.enc4(F.max_pool2d(enc3, (2,2)))
        
        middle = self.middle(F.max_pool2d(enc4, (2,2)))
        
        up1 = torch.cat([enc4, self.up1(middle)], 1)
        dec1 = self.dec1(up1)
        up2 = torch.cat([enc3, self.up2(dec1)], 1)
        dec2 = self.dec2(up2)
        up3 = torch.cat([enc2, self.up3(dec2)], 1)
        dec3 =self.dec3(up3)
        up4 = torch.cat([enc1, self.up4(dec3)], 1)
        dec4 = self.dec4(up4)
        
        final=self.final(dec4)
        
        return final

# WNet
class MyWNet(nn.Module):

    def __init__(self, squeeze, ch_mul=64, in_chans=3, out_chans=1000, padding_mode=None): # 1000 is just a placeholder but idk when its used
        super(MyWNet, self).__init__()
        if out_chans==1000:
            out_chans=in_chans
        self.padding_mode = padding_mode
        self.UEnc=UEnc(squeeze, ch_mul, in_chans, padding_mode)
        self.UDec=UDec(squeeze, ch_mul, out_chans, padding_mode)

    def forward(self, x, returns='dec'):
        enc = self.UEnc(x)
        if returns=='enc':
            return enc
        dec=self.UDec(F.softmax(enc, 1))
        if returns=='dec':
            return dec
        elif returns=='both':
            return enc, dec

# Reconstruction loss
def multichannel_MSE_loss(x, x_prime, weights):
    # MSE loss but with ability to weight loss from each channel differently

    loss = 0
    for channel in range(x.shape[1]):
        mse = nn.MSELoss()(x[:,channel,:,:], x_prime[:,channel,:,:])
        loss += weights[channel]*mse
    loss = loss/(x.shape[1])

    return loss

# Function to train one batch 
def train_op(model, optimizer, input, k, img_size, batch_num, smooth_loss, blob_loss, psi=0.5, device='cpu', train_enc_sup=False, labels=None, freeze_dec=False, target_pos=0, weights=None):

    softmax = nn.Softmax2d()
    smoothLoss = OpeningLoss2D()

    # set to use device
    input = input.to(device) 

    enc = model(input, returns='enc') # predict seg of k="squeeze" classes (NOT n_classes classes)
    if train_enc_sup: # if running supervised (UNet)
        # enc_loss = MSE_loss(enc, labels)
        bp_weight = 5 # dm_weight = 4
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1,1,bp_weight]))
        enc_loss = loss_fn(enc, labels)
    else:
        smooth_wght = 10 if k > 2 else 10 # for binary, smaller smooth loss works better (?)
        n_cut_loss = soft_n_cut_loss(input,  softmax(enc),  img_size)    # from reproduction
        if (smooth_loss and not blob_loss):
            enc_loss = smooth_wght*smoothLoss(softmax(enc)) + 10*n_cut_loss # lets try a bigger weight to the smooth loss (11/8) (was 1e-1 I think)
        if (smooth_loss and blob_loss):
            enc_loss = smooth_wght*smoothLoss(softmax(enc)) + 10*n_cut_loss + 1e-1*blobloss(enc)
            #print(f'smooth loss: {10*smoothLoss(softmax(enc))}, n cut loss: {10*n_cut_loss}, blob loss: {blobloss(enc)}')
        else:
            enc_loss = n_cut_loss
    if torch.isnan(enc_loss).any() == True: 
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(input[-1,target_pos,:,:]) # last img in batch, first channel
        axs[1].imshow(input[-1,1,:,:]) # last img in batch, second channel
        axs[2].imshow(np.argmax(enc[-1,:,:,:].detach().numpy(), axis=0)) # seg for last img in batch
        raise ValueError('enc loss has become NaN')
    enc_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    example_seg = np.argmax(enc[-1,:,:,:].detach().numpy(), axis=0) # seg for last img in batch

    if freeze_dec:
        rec_loss = torch.tensor(np.NaN)
        example_rec = np.zeros_like(example_seg)*np.NaN
        example_rec2 = None
    else:
        dec = model(input, returns='dec') # predict image [INCLUDES ALL CHANNELS]
        rec_loss = multichannel_MSE_loss(input, dec, weights)  # from reprod (MSELoss betwn input and rec imag) BUT with added channel weights
        # rec_loss=torch.mean(torch.pow(torch.pow(input, 2) + torch.pow(dec, 2), 0.5))*(1-psi)
        if torch.isnan(rec_loss).any() == True: raise ValueError('rec loss has become NaN')
        rec_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        example_rec = dec[-1,target_pos,:,:].cpu().detach().numpy() # rec image for last img in batch 
        if dec.shape[1] == 2: # if only two channels, want to plot both
            example_rec2 = dec[-1,1,:,:].cpu().detach().numpy() # rec channel2 for last img in batch 
        else: example_rec2 = None 

    return model, enc_loss, rec_loss, example_seg, example_rec, example_rec2

# Function to train one epoch
def train_WNet(dataloader, wnet, optimizer, k, img_size, WNet_name, smooth_loss, blob_loss, epoch, device='cpu', train_enc_sup=False, freeze_dec=False, target_pos=0, weights=None):
    
    # Create empty lists to collect losses and example segs and recs and corresponding imgs from each batch 
    enc_losses = []
    rec_losses = []
    example_imgs = [] # yes.. could access through dataloader later instead of storing
    example_segs = []
    example_recs = []
    example_img2s = []
    example_rec2s = []

    # Train on each batch in train loader
    for (idx, batch) in enumerate(dataloader):
        print(f'\t   batch {idx}', end='\r')
        
        # Train one batch
        X = batch[0] # batch is [images, labels]
        y = batch[1] # only used if train_enc_sup = True
        wnet, enc_loss, rec_loss, example_seg, example_rec, example_rec2, = train_op(wnet, optimizer, X, k, img_size, smooth_loss=smooth_loss, blob_loss=blob_loss, batch_num=idx, device=device, train_enc_sup=train_enc_sup, labels=y, freeze_dec=freeze_dec, target_pos=target_pos, weights=weights)
        enc_losses.append(enc_loss.detach())
        rec_losses.append(rec_loss.detach())
        example_segs.append(example_seg)
        example_recs.append(example_rec)
        example_imgs.append(X[-1,target_pos,:,:]) # last img in batch, first channel
        if X.shape[1] == 2: # if two channels
            example_rec2s.append(example_rec2)
            example_img2s.append(X[-1,1,:,:])

    # Plot example imgs from each batch 
    cols = 3 if len(example_img2s) == 0 else 5 # if I've stored second rec and image channels
    fig, axs = plt.subplots(len(example_segs), cols, figsize=(cols*3, len(example_segs)*1.5))
    axs[0, 0].set_title('last img, target ch')
    if cols == 3:
        axs[0, 1].set_title('seg (argmax enc)')
        axs[0, 2].set_title('reconstructed [target ch]')
        for i in range(len(example_segs)):
            axs[i,0].set_ylabel(f'Batch {i}')
            axs[i,0].imshow(example_imgs[i], vmin=0, vmax=1) #(X[-1,i,:,:])
            axs[i,1].imshow(example_segs[i])
            axs[i,2].imshow(example_recs[i])
    if cols == 5: 
        axs[0, 1].set_title('last img, 2nd ch')
        axs[0, 2].set_title('seg (argmax enc)')
        axs[0, 3].set_title('rec, target ch')
        axs[0, 4].set_title('rec, 2nd ch')
        for i in range(len(example_segs)):
            axs[i,0].set_ylabel(f'Batch {i}')
            axs[i,0].imshow(example_imgs[i], vmin=0, vmax=1) #(X[-1,i,:,:])
            axs[i,1].imshow(example_img2s[i], vmin=0, vmax=1)
            axs[i,2].imshow(example_segs[i])
            axs[i,3].imshow(example_recs[i])
            axs[i,4].imshow(example_rec2s[i])
    for i in range(len(example_segs)):
        for j in range(cols):
            axs[i,j].xaxis.set_tick_params(labelbottom=False); axs[i,j].yaxis.set_tick_params(labelleft=False); axs[i,j].set_xticks([]); axs[i,j].set_yticks([])
    plt.savefig(f'../NN_storage/{WNet_name}_epoch{epoch}_examples'); plt.close()
    enc_losses.append(torch.mean(torch.FloatTensor(enc_losses)))
    rec_losses.append(torch.mean(torch.FloatTensor(rec_losses)))

    return enc_losses, rec_losses


# Function to Run each validation obs through model, save results. True and predicted vals are saved as 2d maps; e.g. compressed back to original seg format
def save_WNET_results(val_loader, save_dir, model, target_pos=0):

    print(f'Loading model back in, saving results on validation data in {save_dir}')
    if os.path.exists(save_dir) == False: os.mkdir(save_dir)
    i = 0
    for X, y in val_loader:

        X, y = X.to('cpu'), y.to('cpu')
        probs = model(X, returns='enc') # defualt is to return dec, but we want seg
        preds = np.argmax(probs.detach().numpy(), axis=1).astype(float) # turn probabilities [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
        y = np.argmax(y.detach().numpy(), axis=1).astype(np.float) # turn 1-hot-encoded layers [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
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
    '''
    DONT USE THIS - IT DOESNT SEEM TO BE WORKING 
    '''

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
        # preds =  probs_to_preds(probs) # JUST ADDED - WILL IT FIX IT?

        # print(f'preds.shape: {preds.shape}')
        # print(f'targets.shape: {targets.shape}')
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
    
def blobloss(outputs):
    # Metric describing how much the classes group pixels into circular blobs vs long shapes (e.g. outlines of other blobs)
    # Is this sort of forcing it too much?
    # So it seems to just be doing this by shrinking the granule/BP regions - still somewhat choppy edges
    preds = np.argmax(outputs.detach().numpy(), axis=1) # take argmax along classes axis
    edge_pix = 0
    for batch in range(preds.shape[0]):
        cv2.imwrite("temp_img.png", preds[batch,:,:]) 
        edges = np.array(cv2.Canny(cv2.imread("temp_img.png"),0,1.5))
        edge_pix += len(edges[edges > 0])
    loss = edge_pix/(preds.shape[0]*preds.shape[1]**2)
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
        fig, axs = plt.subplots(7, 3, figsize=(3*4, 7*4))
        axs[0,0].set_title('image[0]')
        #ax2.set_title('image[1]')
        axs[0,2].set_title('labels')
        for i in range(7):
            X, y = next(iter(train_loader))
            y = onehot_to_map(y)
            im1 = axs[i,0].imshow(X[0,0,:,:], vmin=0, vmax=1); plt.colorbar(im1, ax=axs[i,0]) # first img in batch, first channel
            if X.shape[1] > 1: im2 = axs[i,1].imshow(X[0,1,:,:], vmin=0, vmax=1); plt.colorbar(im2, ax=axs[i,1]) # first img in batch, first channel
            im3 = axs[i,2].imshow(y[0,:,:]); plt.colorbar(im3, ax=axs[i,2]) # first y in batch, already class-collapsed
        plt.savefig(f'traindata_{name}'); a=b


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

# From https://github.com/AsWali/WNet/blob/master/utils
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

# From https://github.com/AsWali/WNet/blob/master/utils
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

def post_process(preds, data=None):

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

def segment_array_v2(map, resolution, *, skimage_method="li", mark_dim_centers=False, mark_BP=True, bp_min_flux=None, bp_max_size=0.15, footprint=250, pad=None):
    
    """
    NOT EXACTLY THE SAME AS SUNKIT-IMAGE VERSION (DIFFERENT CLASS LABELS, NPY VS MAP, ADDS MARK_FAC FLAG, ADD PAD ARGUEMENT TO ADJUST PAD)
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
    pad: `int`
        Number of pixels to remove from each edge of the image.
        Default is image length / 200

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
    seg_im_fixed = trim_intergranules_v2(segmented_image, mark=mark_dim_centers, pad=pad)
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

def trim_intergranules_v2(segmented_image, mark=False, pad=None):
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
    pad : `int`
        Number of pixels to remove from each edge of the image.
        Default is image length / 200

    Returns
    -------
    segmented_image_fixed : `numpy.ndarray`
        The segmented image without incorrect extra intergranules.
    """

    if len(np.unique(segmented_image)) > 2:
        raise ValueError("segmented_image must only have values of 1 and 0.")
    segmented_image_fixed = np.copy(segmented_image).astype(float)  # Float conversion for correct region labeling.
    # Add padding of IG around edges, because if edges are all GR, will ID all DM as IG
    if pad == None:
        pad = int(np.shape(segmented_image)[0]/100)
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
