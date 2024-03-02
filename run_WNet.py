from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
import torch
import numpy as np
import funclib

# Re-run these all eventually (previously ran them on old segmented I think)!
model_dict = {
    'WNet1':'binary, 1 channel, no smooth loss',
    'WNet2':'4 classes, 1 channel, no smooth loss', 
    'WNet3':'binary, 1 channel, with smooth loss (but bad weighting)',
    'WNet4':'', 
    'WNet5':'3 classes, 2 channels (X, X**3), smooth loss', 
    'WNet6':'3 classes, 4 channels (X, X**3, grad_x, grad_y), smooth loss', 
    'WNet7':'3 classes, 4 channels (X, X**4, grad_x, grad_y), smooth loss', 
    'WNet8':'binary, 2 channels (X, X*2), smooth loss', 
    'WNet9':'', 
    'WNet10':'4 classes, 4 channels (X, X**3, grad_x, grad_y), smooth loss',
    'WNet10dv2':'4 classes, 4 channels (X, X**3, grad_x, grad_y), smooth loss', 
    'WNet11':'4 classes, 1 channel, smooth loss',
    'WNet12':'3 classes, 1 channel, smooth loss',
    'FreezeNet1': '1 epoch sup, 2 epoch unsup, 3 classes, 1 channels, smooth loss',
    'FreezeNet2': '1 epoch sup, 2 epoch unsup, 3 classes, 2 channels (X, X*2), smooth loss',
    'FreezeNet3': '2 epoch sup, NO unsup, 3 classes, 2 channels (X, X*2), smooth loss',
    'FreezeNet4': '1 epoch sup W/O decoder, 2 epoch unsup, 3 classes, 2 channels (X, X*2), smooth loss',
    'WNet13': '4 classes, 3 channels (X, X**3, smoothed), smooth loss'}

WNet_name = 'WNet2m'
n_classes = 3
channels = ['power2']
imdir = "../Data/UNetData_MURaM/images/" 
segdir = "../Data/UNetData_MURaM/seg_images/"
in_channels = len(channels)+1
im_size = 128  # [9, 18, 36, 72, 144] or [8, 16, 32, 64, 128]
randomSharp = False
smooth_loss = False
# bp_weight = 5; dm_weight = 4
load_model = False
num_epochs = 3 
num_sup = 0
freeze_dec = False # freeze decoder during sup training of encoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the data
batch_size = 16
train_ds = funclib.MyDataset(image_dir=f"{imdir}train", mask_dir=f"{segdir}train", norm=True, n_classes=n_classes, channels=channels, randomSharp=randomSharp, im_size=im_size) # multichannel=True, channels=['deltaBinImg'], 
train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True)
val_ds = funclib.MyDataset(image_dir=f"{imdir}val", mask_dir=f"{segdir}val", norm=True, n_classes=n_classes, channels=channels, randomSharp=randomSharp, im_size=im_size) # multichannel=True, channels=['deltaBinImg'],
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False)
funclib.check_inputs(train_ds, train_loader, savefig=False, name=WNet_name)

# Define model, optimizer, and transforms
# squeeze (k) is "classes" in seg predicted by dec - why would we ever not want this to be n_classes? some source says paper probabaly uses k=20, but they are doing binary predictions...????
# out_channels is channels for the final img (not classes for seg), so want same as in_channels, right?
model = funclib.MyWNet(squeeze=n_classes, ch_mul=64, in_chans=in_channels, out_chans=in_channels).to(device)
learning_rate = 0.003
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Run for every epoch
n_cut_losses_avg = []
rec_losses_avg = []
print('Training')
for epoch in range(num_epochs):

    train_enc_sup = True if epoch < num_sup else False
    if epoch >= num_sup: freeze_dec = False
    print(f'\tEpoch {epoch}, ({f"supervised, freeze_dec={freeze_dec}" if train_enc_sup==True else f"unsupervised, freeze_dec={freeze_dec}"})')

    # Train (returning losses)
    enc_losses, rec_losses = funclib.train_WNet(train_loader, model, optimizer, k=n_classes, img_size=(im_size, im_size), WNet_name=WNet_name, smooth_loss=smooth_loss, epoch=epoch,  device=device, train_enc_sup=train_enc_sup, freeze_dec=freeze_dec)

    # # check accuracy 
    # accuracy, dice_score = validate(val_loader, model)
    # print(f"\tGot accuracy {accuracy:.2f} and dice score: {dice_score/len(val_loader)}")
    # model.train() # set model back into train mode

# Add losses to avg losses
n_cut_losses_avg.append(torch.mean(torch.FloatTensor(enc_losses)))
rec_losses_avg.append(torch.mean(torch.FloatTensor(rec_losses)))

# images, labels = next(iter(dataloader))
# enc, dec = wnet(images)

# Save model 
torch.save(model.state_dict(), f'../NN_storage/{WNet_name}.pth')
print(f'Saving trained model as {WNet_name}.pth, and saving average losses')
np.save(f'../NN_outputs/{WNet_name}_n_cut_losses', n_cut_losses_avg)
np.save(f'../NN_outputs/{WNet_name}_rec_losses', rec_losses_avg)

# Load it back in and save results on validation data 
model = funclib.MyWNet(squeeze=n_classes, ch_mul=64, in_chans=in_channels, out_chans=in_channels)
model.load_state_dict(torch.load(f'../NN_storage/{WNet_name}.pth'))
funclib.save_WNET_results(val_loader, save_dir=f'../NN_outputs/{WNet_name}_outputs', model=model)