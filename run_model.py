import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
from utils import run_utils, data_utils, models
import argparse
import json
import os, shutil

'''
Script to train a supervised model. Use for predicting magnetic flux from photometry.
'''

def run_model(d, gpu, test_only=False):
    
    # Set out and data dirs
    net_name = d['net_name'] 
    outdir = "../model_runs_mag" # create this before running
    imgdir = d['img_dir'] # phot images (if predicting mag from phot)
    truedir = d['true_dir'] # mag images (if predicting mag from phot)
    if not os.path.exists(outdir) and os.path.exists('../'+outdir): # if running from Solar_Segmentation/analysis/unsupervised.ipynb notebook for debugging 
        outdir='../'+outdir
    exp_outdir = f'{outdir}/{d["sub_dir"]}/{net_name}/'
    
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
    train_ds = data_utils.dataset(image_dir=imgdir, mask_dir=truedir, set='train', norm=False, channels=channels, randomSharp=d['randomSharp'], im_size=im_size) # multichannel=True, channels=['deltaBinImg'], 
    train_loader = DataLoader(train_ds, batch_size=d['batch_size'], pin_memory=True, shuffle=True)
    test_ds = data_utils.dataset(image_dir=imgdir, mask_dir=truedir, set='val', norm=False, channels=channels, randomSharp=d['randomSharp'], im_size=im_size) # multichannel=True, channels=['deltaBinImg'],
    test_loader = DataLoader(test_ds, batch_size=d['batch_size'], pin_memory=True, shuffle=False)
    data_utils.check_inputs(train_ds, train_loader, savefig=False, name=net_name)
    
    # Define model and loss func optimizer
    device = torch.device('cuda') if eval(str(gpu)) else torch.device('cpu') #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if d['model'] == 'magNet0':
        model = models.magNet0(in_channels=in_channels, hidden_channels=d['hidden_channels']).to(device)
    if d['model'] == 'regVGG':
        model = models.regVGG(in_channels=in_channels).to(device)
    if d['model'] == 'DeepVel':
        model = models.DeepVel(in_channels=in_channels).to(device)
    loss_func = run_utils.get_loss_func(d['loss_func']) 
    optimizer = torch.optim.SGD(model.parameters(), lr=d["learning_rate"])
    
    # Create outdir and train 
    if not eval(str(test_only)):

        # Train model
        losses = []
        print(f"Training {net_name} (training on {device})", flush=True)
        for epoch in range(d['num_epochs']):
            print(f'Epoch {epoch}', flush=True)
            save_examples = True if d['num_epochs'] < 4 or epoch % 2 == 0 or epoch == d['num_epochs']-1 or epoch == d['num_epochs']-2 else False # if more than 5 epochs, save imgs for only every other epoch, 2nd to last, and last epoch 
            loss = run_utils.train_Net(train_loader, model, loss_func, loss_name=d['loss_func'], optimizer=optimizer, chan_weights=d['weights'], device=device, save_examples=save_examples, save_dir=exp_outdir, epoch=epoch)
            losses.append(loss.cpu().detach().numpy())

        # Save model 
        torch.save(model.state_dict(), f'{exp_outdir}/{net_name}.pth')
        print(f'Saving trained model as {exp_outdir}/{net_name}.pth, and saving average losses', flush=True)
        np.save(f'{exp_outdir}/losses', np.array(losses))

    # Load it back in and save results on test data 
    if d['model'] == 'magNet0':
        model = models.magNet0(in_channels=in_channels, hidden_channels=d['hidden_channels'])
    if d['model'] == 'regVGG':
        model = models.regVGG(in_channels=in_channels).to(device)
    if d['model'] == 'DeepVel':
        model = models.DeepVel(in_channels=in_channels).to(device)
    model.load_state_dict(torch.load(f'{exp_outdir}/{net_name}.pth'))
    test_outdir = f'{exp_outdir}/test_preds_MURaM' if 'MURaM' in d['img_dir'] else f'{exp_outdir}/test_preds_DKIST'
    run_utils.save_model_results(test_loader, save_dir=test_outdir, model=model, device=device)


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
        print(f'RUNNING EXPERIMENT {d["net_name"]} \nexp dict: {d}')
        run_model(d, args.gpu)
        print(f'DONE')
    print('FINISHED ALL EXPERIMENTS')
