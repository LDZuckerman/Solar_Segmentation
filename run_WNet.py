import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
from utils import run_utils, data_utils, models
import argparse
import json
import os, shutil

def run_wnet_model(d, gpu, test_only=False):
    
    # Set out and data dirs
    WNet_name = d['WNet_name'] 
    WNet_id = d['WNet_name'].replace('WNet', '')
    outdir = f"../model_runs_seg" 
    imgdir = d['img_dir']
    segdir = d['seg_dir']
    if not os.path.exists('../model_runs_seg'): # if running from Solar_Segmentation/analysis/unsupervised.ipynb notebook for debugging
        outdir='../'+outdir
    exp_outdir = f'{outdir}/{d["sub_dir"]}/{WNet_name}/'
    
    # Copy exp dict file for convenient future reference and create exp outdir 
    if not os.path.exists(exp_outdir): 
        print(f'Creating experiment output dir {exp_outdir}')
        os.makedirs(exp_outdir)
    elif not test_only:
        print(f'Experiment output dir {exp_outdir} already exists - contents will be overwritten')
    if not test_only:
        print(f'Copying exp dict into {exp_outdir}/exp_file.json')
        json.dump(d, open(f'{exp_outdir}/exp_file.json','w'))
    
    # Get data (note: truth data not used in training of WNet, just loaded to easily store results)
    print(f"Loading data from {imgdir}", flush=True)
    channels = d['channels'] 
    im_size = d['img_size']
    train_ds = data_utils.dataset(image_dir=imgdir, mask_dir=segdir, set='train', norm=False, n_classes=d['n_classes'], channels=channels, randomSharp=d['randomSharp'], im_size=im_size, add_mag=d['reconstruct_mag'], inject_brightmasks=d['inject_brightmasks']) # multichannel=True, channels=['deltaBinImg'], 
    train_loader = DataLoader(train_ds, batch_size=d['batch_size'], pin_memory=True, shuffle=True)
    test_ds = data_utils.dataset(image_dir=imgdir, mask_dir=segdir, set='val', norm=False, n_classes=d['n_classes'], channels=channels, randomSharp=d['randomSharp'], im_size=im_size) # multichannel=True, channels=['deltaBinImg'],
    test_loader = DataLoader(test_ds, batch_size=d['batch_size'], pin_memory=True, shuffle=False)
    data_utils.check_inputs(train_ds, train_loader, savefig=True, name=WNet_name, reconstruct_mag=d['reconstruct_mag'])
    
    # Define model
    device = torch.device('cuda') if eval(str(gpu)) else torch.device('cpu') # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if eval(str(d['reconstruct_mag'])): # last "channel" is just flux that is tacked on 
        print(f'reconstruct_mag is True')
        in_channels = int(channels[0][channels[0].find('_')+1:]) if channels[0].startswith('time')-1 else len(channels)-1
        out_channels =  in_channels + 1
    else:
        in_channels = int(channels[0][channels[0].find('_')+1:]) if channels[0].startswith('time') else len(channels)
        out_channels = in_channels
    target_pos = int(np.floor(in_channels/2)) if channels[0].startswith('time') else 0 # position of target within channels axis 
    if str(d['pretrained_dec']) != 'None':
        if not os.path.exists(f'../model_runs_dec/MURaM/{d["pretrained_dec"]}'):
            raise ValueError(f'Pre-trained decoder ../model_runs_dec/MURaM/{d["pretrained_dec"]} does not exist.')
        else:
            print(f'Reading in pre-trained decoder {d["pretrained_dec"]}. This will be used to initialize the decoder wieghts.')
            dec_channels = json.load(open(f'../model_runs_dec/MURaM/{d["pretrained_dec"]}/exp_file.json'))['channels']
            if d['channels'] != dec_channels:
                raise ValueError(f'Channels expected by this model ({d["channels"]}) not the same as channels that indicated dec has been trained to reproduce ({dec_channels})')
    model = models.WNet(in_chans=in_channels, n_classes=d['n_classes'], out_chans=out_channels, dec_depth=d['dec_depth'], double_dec=eval(str(d['double_dec'])), dec_ch_mul=64, kernel_size=d['kernel_size'], padding_mode=d['padding_mode'], reconstruct_mag=eval(str(d['reconstruct_mag'])), pretrained_dec=d['pretrained_dec'], activation=eval(str(d['activation']))).to(device)                                                                                                                                    
    # Create outdir and train 
    if not eval(str(test_only)):

        # Train model
        optimizer = torch.optim.SGD(model.parameters(), lr=d["learning_rate"])
        n_cut_losses_avg = []
        rec_losses_avg = []
        print(f"Training {WNet_name} (training on {device})", flush=True)
        for epoch in range(d['num_epochs']):
            print(f'Epoch {epoch}', flush=True)
            save_examples = True if d['num_epochs'] < 4 or epoch % 2 == 0 or epoch == d['num_epochs']-1 or epoch == d['num_epochs']-2 else False # if more than 5 epochs, save imgs for only every other epoch, 2nd to last, and last epoch 
            enc_losses, rec_losses = run_utils.train_WNet(train_loader, model, optimizer, k=d['n_classes'], img_size=(d['img_size'], d['img_size']), exp_outdir=exp_outdir, WNet_id=WNet_id, smooth_wght=d['smooth_wght'], cohesion_wght=d['chsn_wght'], ncut_wght=d['ncut_wght'], cont_wght=d['cont_wght'], epoch=epoch,  device=device, target_pos=target_pos, weights=d['weights'], reconstruct_mag=d['reconstruct_mag'], save_examples=save_examples, temp_dir=exp_outdir)
            n_cut_losses_avg.append(torch.mean(torch.FloatTensor(enc_losses)))
            rec_losses_avg.append(torch.mean(torch.FloatTensor(rec_losses)))

        # Save model 
        torch.save(model.state_dict(), f'{exp_outdir}/{WNet_name}.pth')
        print(f'Saving trained model as {exp_outdir}/{WNet_name}.pth, and saving average losses', flush=True)
        np.save(f'{exp_outdir}/n_cut_losses', n_cut_losses_avg)
        np.save(f'{exp_outdir}/rec_losses', rec_losses_avg)

    # Load it back in and save results on test data 
    model = models.WNet(in_chans=in_channels, n_classes=d['n_classes'], out_chans=out_channels, dec_depth=d['dec_depth'], double_dec=eval(str(d['double_dec'])), dec_ch_mul=64, kernel_size=d['kernel_size'], padding_mode=d['padding_mode'], reconstruct_mag=eval(str(d['reconstruct_mag'])), pretrained_dec=d['pretrained_dec'], activation=eval(str(d['activation']))).to(device)                                                                                                                                    
    print(f'Loading model back in from {exp_outdir}/{WNet_name}.pth and testing')
    model.load_state_dict(torch.load(f'{exp_outdir}/{WNet_name}.pth'))
    test_outdir = f'{exp_outdir}/test_preds_MURaM' if 'MURaM' in d['img_dir'] else f'{exp_outdir}/test_preds_DKIST'
    run_utils.save_WNET_results(test_loader, save_dir=test_outdir, model=model, target_pos=target_pos, device=device)


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
        print(f'RUNNING EXPERIMENT {d["WNet_name"]} \nexp dict: {d}')
        run_wnet_model(d, args.gpu)
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