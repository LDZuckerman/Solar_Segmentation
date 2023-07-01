import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
from PIL import Image
from torch.utils.data import Dataset, TensorDataset, DataLoader
from albumentations.pytorch import ToTensorV2
import torchvision 
import torchvision.transforms.functional as TF
from tqdm import tqdm
import torch.optim as optim
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import funclib

# Dataset
class MyDataset(Dataset):

    def __init__(self, image_dir, mask_dir, norm=False, multichannel=False, channels=[], multiclass=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.norm = norm
        self.multichannel = multichannel
        self.channels = channels
        self.multiclass = multiclass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index]) # path to one data image
        mask_path = os.path.join(self.mask_dir, 'SEG_'+self.images[index]) # path to one labels image THIS SHOULD ENSURE ITS THE MASK CORRESPONDING TO THE CORRECT IMAGE
        img = np.load(img_path).newbyteorder().byteswap() 
        if self.norm:  # normalize 
            img = (img - np.mean(img))/np.std(img)
        if self.multichannel: # Add feature layers
            image = np.zeros((len(self.channels)+1, img.shape[0], img.shape[1]), dtype=np.float32) # needs to be float32 not float64
            image[0, :, :] = img
            for i in range(len(self.channels)):
                image[i+1, :, :] = funclib.get_feature(img, self.channels[i], index)
        else: # Add dummy axis
            image = np.zeros((1, img.shape[0], img.shape[1]), dtype=np.float32) # needs to be float32 not float64
            image[0, :, :] = img
        labels = np.load(mask_path).newbyteorder().byteswap() 
        if self.multiclass: # One-hot encode targets so they are the correct size
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
        else: # Add dummy axis
            mask = np.zeros((1, labels.shape[0], labels.shape[1]), dtype=np.float32) # needs to be float32 not float64
            mask[0, :, :] = labels
        return image, mask


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
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


# Function to train one epoch
def train(loader, model, optimizer, loss_fn, scaler, multiclass=False, dm_weight=1, bp_weight=1):
    
    loop = tqdm(loader) # is this a progress bar? but then below its looping through loop

    # Train on each set in train loader
    for batch_idx, (data, targets) in enumerate(loop):
        # set to use cpu
        data = data.to(device="cpu")
        targets = targets.float().to(device='cpu')
        # forward
        predictions = model(data) # call model to get predictions (not class probs; contains negs; apply sigmoid to get probs).
        if isinstance(loss_fn, funclib.multiclass_MSE_loss):
            loss = loss_fn(predictions, targets, dm_weight, bp_weight) # compute loss
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
def validate(val_loader, model):

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval() # set model into eval mode
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to("cpu")
            y = y.to("cpu")#.unsqueeze(1)
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
def save_val_results(val_loader, save_dir):
    print(f'Loading model back in, saving results on validation data in {save_dir}')
    if os.path.exists(save_dir) == False: os.mkdir(save_dir)
    i = 0
    for X, y in val_loader:
        X, y = X.to('cpu'), y.to('cpu')
        preds = torch.sigmoid(model(X))
        if preds.shape[1] == 1: # if binary (predictions have 1 layer)
            preds = (preds > 0.5).float() # turn regression value (between zero and 1, e.g. "prob of being 1") into predicted 0s and 1s
        else: # if muliclasss (predictions have n_classes layers)
            preds = np.argmax(preds.detach().numpy(), axis=1) # turn probabilities [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
            y = np.argmax(y.detach().numpy(), axis=1) # turn 1-hot-encoded layers [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
            preds = preds/2 # change from idx to class value 0 -> 0 (IG), 1 -> 0.5 (DM), 2 -> 1 (G), 3 -> 1.5 (BP)
            y = y/2 # change from idx to class value 0 -> 0 (IG), 1 -> 0.5 (DM), 2 -> 1 (G), 3 -> 1.5 (BP)
        for j in range(np.shape(preds)[0]):
            np.save(f'{save_dir}/x_{i}', np.array(X[j]))
            np.save(f'{save_dir}/true_{i}', np.array(y[j]))
            np.save(f'{save_dir}/pred_{i}', np.array(preds[j]))
            i += 1

# Get the data, applying some transformations (Should pull segs and truths correctly based on their positions in the directories)
batch_size = 16
train_ds = MyDataset(image_dir="../Data/UNetData_v2/images/train", mask_dir="../Data/UNetData_v2/seg_images/train",multiclass=True) # multichannel=True, channels=['deltaBinImg'], 
train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True)
val_ds = MyDataset(image_dir="../Data/UNetData_v2/images/val", mask_dir="../Data/UNetData_v2/seg_images/val", multiclass=True) # multichannel=True, channels=['deltaBinImg'],
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False)
funclib.check_inputs(train_ds, train_loader)

# Define model (as an instance of MyNeuralNet), loss function, and optimizer
model = MyUNet(in_channels=1, out_channels=4).to("cpu")
loss_fn = funclib.multiclass_MSE_loss() 
optimizer = optim.Adam(model.parameters(), lr=1e-4)
load_model = False
if load_model: 
    model.load_state_dict(torch.load("../NN_storage/UNET_checkpoint.pth.tar")["state_dict"])

# Train
print('Training:')
num_epochs = 3
scaler = torch.cuda.amp.GradScaler() # Don't have cuda
for epoch in range(num_epochs):
    print(f'\tEpoch {epoch}')

    # Train and save snapshot of this epoch's training, in case it crashes while training next one
    train(train_loader, model, optimizer, loss_fn, scaler, dm_weight=10, bp_weight=10) # call model
    state = {"state_dict": model.state_dict(), "optimizer":optimizer.state_dict(),}
    torch.save(state, "../NN_storage/UNET_checkpoint.pth.tar"); print(f'\tSaving checkpoint to ../NN_storage/UNET_checkpoint.pth.tar')

    # check accuracy 
    accuracy, dice_score = validate(val_loader, model)
    print(f"\tGot accuracy {accuracy:.2f} and dice score: {dice_score/len(val_loader)}")
    model.train() # set model back into train mode

# Save model 
torch.save(model.state_dict(), '../NN_storage/UNet7.pth')
print('Saving trained model as UNet7.pth')

# Load it back in and save results on validation data 
model = MyUNet(in_channels=2, out_channels=4)
model.load_state_dict(torch.load('../NN_storage/UNet7.pth'))
save_val_results(val_loader, save_dir='../UNet7_outputs')
