import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
from utils import run_utils, data_utils, models
import argparse
import json
import os, shutil

'''
Script to train a UNet model
Currently for magentic flux prediction, although could be used for supervised segmentation
'''

def run_unet_model(d, gpu, test_only=False):
    
    # Set out and data dirs
    UNet_name = d['UNet_name'] 
    outdir = f"../model_runs_mag" # create this before running
    imgdir = d['img_dir']
    truedir = d['true_dir'] # mag images
    if not os.path.exists(outdir) and os.path.exists('../'+outdir): # if running from Solar_Segmentation/analysis/unsupervised.ipynb notebook for debugging 
        outdir='../'+outdir
    exp_outdir = f'{outdir}/{d["sub_dir"]}/{UNet_name}/'
    
    # Copy exp dict file for convenient future reference and create exp outdir 
    if not os.path.exists(exp_outdir): 
        print(f'Creating experiment output dir {os.getcwd()}/{exp_outdir}')
        os.makedirs(exp_outdir)
    elif not test_only:
        print(f'Experiment output dir {os.getcwd()}/{exp_outdir} already exists - contents will be overwritten')
    print(f'Copying exp dict into {os.getcwd()}/{exp_outdir}/exp_file.json')
    json.dump(d, open(f'{exp_outdir}/exp_file.json','w'))
    
    # Get data 
    print(f"Loading data from {imgdir} and {truedir}", flush=True)
    channels = d['channels'] 
    im_size = 128
    in_channels = int(channels[0][channels[0].find('_')+1:]) if channels[0].startswith('time') else len(channels)
    target_pos = int(np.floor(in_channels/2)) if channels[0].startswith('time') else 0 # position of target within channels axis
    train_ds = data_utils.dataset(image_dir=imgdir, mask_dir=truedir, set='train', norm=False, n_classes=d['n_classes'], channels=channels, randomSharp=d['randomSharp'], im_size=im_size) # multichannel=True, channels=['deltaBinImg'], 
    train_loader = DataLoader(train_ds, batch_size=d['batch_size'], pin_memory=True, shuffle=True)
    test_ds = data_utils.dataset(image_dir=imgdir, mask_dir=truedir, set='val', norm=False, n_classes=d['n_classes'], channels=channels, randomSharp=d['randomSharp'], im_size=im_size) # multichannel=True, channels=['deltaBinImg'],
    test_loader = DataLoader(test_ds, batch_size=d['batch_size'], pin_memory=True, shuffle=False)
    data_utils.check_inputs(train_ds, train_loader, savefig=False, name=UNet_name)
    
    # Define model and loss func optimizer
    device = torch.device('cuda') if eval(str(gpu)) else torch.device('cpu') #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'mag' in UNet_name:
        model = models.magNet(in_channels=in_channels).to(device)
    loss_func = run_utils.get_loss_func(d['loss_func']) # [8/12/24] was I re-initializing this each time time I computed loss before???
    optimizer = torch.optim.SGD(model.parameters(), lr=d["learning_rate"])
    
    # Create outdir and train 
    if not eval(str(test_only)):

        # Train model
        losses = []
        print(f"Training {UNet_name} (training on {device})", flush=True)
        for epoch in range(d['num_epochs']):
            print(f'Epoch {epoch}', flush=True)
            save_examples = True if d['num_epochs'] < 4 or epoch % 2 == 0 or epoch == d['num_epochs']-1 or epoch == d['num_epochs']-2 else False # if more than 5 epochs, save imgs for only every other epoch, 2nd to last, and last epoch 
            loss = run_utils.train_Net(train_loader, model, loss_func, optimizer, bp_weight=d['bp_wght'], save_examples=save_examples, save_dir=exp_outdir, epoch=epoch)#train_loader, model, optimizer, k=d['n_classes'], img_size=(d['img_size'], d['img_size']), exp_outdir=exp_outdir, UNet_id=UNet_id, epoch=epoch,  device=device, target_pos=target_pos, weights=d['weights'], save_examples=save_examples)
            losses.append(loss.cpu().detach().numpy())

        # Save model 
        torch.save(model.state_dict(), f'{exp_outdir}/{UNet_name}.pth')
        print(f'Saving trained model as {exp_outdir}/{UNet_name}.pth, and saving average losses', flush=True)
        np.save(f'{exp_outdir}/losses', np.array(losses))

    # Load it back in and save results on test data 
    if 'mag' in UNet_name:
        model = models.magNet(in_channels=in_channels)
    model.load_state_dict(torch.load(f'{exp_outdir}/{UNet_name}.pth'))
    test_outdir = f'{exp_outdir}/test_preds_MURaM' if 'MURaM' in d['img_dir'] else f'{exp_outdir}/test_preds_DKIST'
    if 'mag' in UNet_name:
        run_utils.save_magNET_results(test_loader, save_dir=test_outdir, model=model)
    else:
        run_utils.save_UNET_results(test_loader, save_dir=test_outdir, model=model, target_pos=target_pos)


if __name__ == "__main__":
    # Read in arguements
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--f", type=str, required=True)
    parser.add_argument("-gpu", "--gpu", type=str, required=True)
    args = parser.parse_args()
    
    # Iterate through experiments (or single experiment)
    with open(args.f) as file:
        exp_dicts = json.load(file)
    if isinstance(exp_dicts, dict): # If experiment file is a single dict, not a list of dicts
        exp_dicts = [exp_dicts]
    for d in exp_dicts:
        print(f'RUNNING EXPERIMENT {d["UNet_name"]} \nexp dict: {d}')
        run_unet_model(d, args.gpu)
        print(f'DONE')
    print('FINISHED ALL EXPERIMENTS')


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