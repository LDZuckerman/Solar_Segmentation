from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
import torch
import numpy as np
from utils import funclib
import argparse
import json
import os, shutil

# Read in arguements
parser = argparse.ArgumentParser()
parser.add_arguement("-f", "--f", type=str, required=True)
args = parser.parse_args()

# Load construction dictionary from exp file
d = json.load(open(args.f, 'rb'))
WNet_name = d['Name'] 
outdir = f"../WNET_runs/{WNet_name}/" 
if not os.path.exists(outdir): 
    os.makedirs(outdir)

# Copy exp dict file for convenient future reference
listfile = '../WNet_runs/exp_dicts.json'
if not os.path.exists(listfile):
    os.makedirs(listfile)
    exp_list = []
else:
    exp_list = json.load(open(listfile, 'rb'))
exp_list.append(d)
json.dump(exp_list, listfile)

# Get data # TRUTH DATA NOT USED, SO SHOULD REMOVE FROM LOAD-IN
print(f"Loading data from {d['img_dir']}")
channels = d['channels'] 
in_channels = int(channels[0][channels[0].find('_')+1:]) if channels[0].startswith('time') else len(channels)
target_pos = int(np.floor(in_channels/2)) if channels[0].startswith('time') else 0 # position of target within channels axis
train_ds = funclib.MyDataset(image_dir=d['img_dir'], mask_dir=d['truth_dir'], set='train', norm=False, n_classes=d['n_classes'], channels=channels, randomSharp=d['randomSharp'], im_size=d['im_size']) # multichannel=True, channels=['deltaBinImg'], 
train_loader = DataLoader(train_ds, batch_size=d['batch_size'], num_workers=2, pin_memory=True, shuffle=True)
test_ds = funclib.MyDataset(image_dir=d['img_dir'], mask_dir=d['truth_dir'], set='val', norm=False, n_classes=d['n_classes'], channels=channels, randomSharp=d['randomSharp'], im_size=d['im_size']) # multichannel=True, channels=['deltaBinImg'],
test_loader = DataLoader(test_ds, batch_size=d['batch_size'], num_workers=2, pin_memory=True, shuffle=False)
funclib.check_inputs(train_ds, train_loader, savefig=False, name=WNet_name)

# Define model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = funclib.MyWNet(squeeze=d['n_classes'], ch_mul=64, in_chans=in_channels, out_chans=in_channels, padding_mode=d['padding_model']).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=d["learning_rate"])

# Run for every epoch
n_cut_losses_avg = []
rec_losses_avg = []
print(f"Training WNet {WNet_name}")
for epoch in range(d['num_epochs']):
    enc_losses, rec_losses = funclib.train_WNet(train_loader, model, optimizer, k=d['n_classes'], img_size=(d['img_size'], d['img_size']), WNet_name=WNet_name, smooth_loss=d['smooth_loss'] , blob_loss=d['blob_loss'], epoch=epoch,  device=device, train_enc_sup=False, freeze_dec=False, target_pos=target_pos, weights=d['weights'])
    n_cut_losses_avg.append(torch.mean(torch.FloatTensor(enc_losses)))
    rec_losses_avg.append(torch.mean(torch.FloatTensor(rec_losses)))

# Save model 
torch.save(model.state_dict(), f'{outdir}/{WNet_name}.pth')
print(f'Saving trained model as {outdir}/{WNet_name}.pth, and saving average losses')
np.save(f'{outdir}/{WNet_name}_n_cut_losses', n_cut_losses_avg)
np.save(f'{outdir}/{WNet_name}_rec_losses', rec_losses_avg)

# Load it back in and save results on test data 
model = funclib.MyWNet(squeeze=d['n_classes'], ch_mul=64, in_chans=in_channels, out_chans=in_channels, padding_mode=d['padding_mode'])
model.load_state_dict(torch.load(f'{outdir}/{WNet_name}.pth'))
funclib.save_WNET_results(test_loader, save_dir=f'{outdir}/{WNet_name}_outputs', model=model, target_pos=target_pos)



# import funclib
# import importlib
# importlib.reload(funclib)

# WNet_name = 'WNet35nm'
# n_classes = 3 #
# channels = ['X', 'median_residual'] # ['X', 'Bz']# ['timeseries40_5'] # ['X', 'Bz'] # ['timeseries20_5'] # ['timeseries40_9'] # ['X', 'power2', 'binary_residual'] 
# weights = [1, 4] #[1, 4] # weight channels differently in the rec loss (mse loss)
# # imdir = '../Data/UNetData_v2_subset/norm_images/' # "../Data/UNetData_MURaM/images/" # "../Data/UNetData_DKIST_TSeries/images/" #"../Data/UNetData_MURaM_TSeries40/images/" 
# # segdir =  '../Data/UNetData_v2_subset/seg_images/' # "../Data/UNetData_MURaM/seg_images/" # "../Data/UNetData_DKIST_TSeries/seg_images/" #"../Data/UNetData_MURaM_TSeries40/seg_images/" # why did i use full UNetData, and not v2, for the other WNets? MURaM is v2
# imdir =  "../Data/UNetData_MURaM/norm_images/" 
# segdir =  "../Data/UNetData_MURaM/seg_images/" 
# im_size = 128  # [5, 10, 20, 40, 80, 160] or [4, 8, 16, 32, 64, 128]
# randomSharp = False
# smooth_loss = True
# blob_loss = False
# padding_mode = 'reflect'
# load_model = False
# num_epochs = 3 
# num_sup = 0 
# freeze_dec = False # freeze decoder during sup training of encoder
# in_channels = int(channels[0][channels[0].find('_')+1:]) if channels[0].startswith('time') else len(channels)
# target_pos = int(np.floor(in_channels/2)) if channels[0].startswith('time') else 0 # position of target within channels axis
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Get the data
# batch_size = 16
# print(WNet_name)
# train_ds = funclib.MyDataset(image_dir=imdir, mask_dir=segdir, set='train', norm=False, n_classes=n_classes, channels=channels, randomSharp=randomSharp, im_size=im_size) # multichannel=True, channels=['deltaBinImg'], 
# train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True)
# val_ds = funclib.MyDataset(image_dir=imdir, mask_dir=segdir, set='val', norm=False, n_classes=n_classes, channels=channels, randomSharp=randomSharp, im_size=im_size) # multichannel=True, channels=['deltaBinImg'],
# val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False)
# funclib.check_inputs(train_ds, train_loader, savefig=False, name=WNet_name)

# # Define model, optimizer, and transforms
# # squeeze (k) is "classes" in seg predicted by dec - why would we ever not want this to be n_classes? some source says paper probabaly uses k=20, but they are doing binary predictions...????
# # out_channels is channels for the final img (not classes for seg), so want same as in_channels, right?
# model = funclib.MyWNet(squeeze=n_classes, ch_mul=64, in_chans=in_channels, out_chans=in_channels, padding_mode=padding_mode).to(device)
# learning_rate = 0.003
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# # Run for every epoch
# n_cut_losses_avg = []
# rec_losses_avg = []
# print('Training')
# for epoch in range(num_epochs):

#     train_enc_sup = True if epoch < num_sup else False
#     if epoch >= num_sup: freeze_dec = False
#     print(f'\tEpoch {epoch}, ({f"supervised, freeze_dec={freeze_dec}" if train_enc_sup==True else f"unsupervised, freeze_dec={freeze_dec}"})')

#     # Train (returning losses)
#     enc_losses, rec_losses = funclib.train_WNet(train_loader, model, optimizer, k=n_classes, img_size=(im_size, im_size), WNet_name=WNet_name, smooth_loss=smooth_loss, blob_loss=blob_loss, epoch=epoch,  device=device, train_enc_sup=train_enc_sup, freeze_dec=freeze_dec, target_pos=target_pos, weights=weights)

#     # # check accuracy 
#     # accuracy, dice_score = validate(val_loader, model)
#     # print(f"\tGot accuracy {accuracy:.2f} and dice score: {dice_score/len(val_loader)}")
#     # model.train() # set model back into train mode

# # Add losses to avg losses
# n_cut_losses_avg.append(torch.mean(torch.FloatTensor(enc_losses)))
# rec_losses_avg.append(torch.mean(torch.FloatTensor(rec_losses)))

# # images, labels = next(iter(dataloader))
# # enc, dec = wnet(images)

# # Save model 
# torch.save(model.state_dict(), f'../NN_storage/{WNet_name}.pth')
# print(f'Saving trained model as {WNet_name}.pth, and saving average losses')
# np.save(f'../NN_outputs/{WNet_name}_n_cut_losses', n_cut_losses_avg)
# np.save(f'../NN_outputs/{WNet_name}_rec_losses', rec_losses_avg)

# # Load it back in and save results on validation data 
# model = funclib.MyWNet(squeeze=n_classes, ch_mul=64, in_chans=in_channels, out_chans=in_channels, padding_mode=padding_mode)
# model.load_state_dict(torch.load(f'../NN_storage/{WNet_name}.pth'))
# funclib.save_WNET_results(val_loader, save_dir=f'../NN_outputs/{WNet_name}_outputs', model=model, target_pos=target_pos)