from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
import torch
import funclib

UNet_name = 'UNet9'
multiclass = True
channels = []
imdir = "../Data/UNetData_v2_subset/images/"
segdir = "../Data/UNetData_v2_subset/seg_images/"
in_channels = 1
n_classes = 4
loss_fn = funclib.multiclass_MSE_loss() 
load_model = False
num_epochs = 3 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the data, applying some transformations (Should pull segs and truths correctly based on their positions in the directories)
batch_size = 16
train_ds = funclib.MyDataset(image_dir=f"{imdir}train", mask_dir=f"{segdir}train", n_classes=n_classes, channels=channels) # multichannel=True, channels=['deltaBinImg'], 
train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True)
val_ds = funclib.MyDataset(image_dir=f"{imdir}val", mask_dir=f"{segdir}val", n_classes=n_classes, channels=channels) # multichannel=True, channels=['deltaBinImg'],
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False)
funclib.check_inputs(train_ds, train_loader)

# Define model (as an instance of MyNeuralNet), loss function, and optimizer
model = funclib.MyUNet(in_channels, n_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
if load_model: 
    model.load_state_dict(torch.load("../NN_storage/UNET_checkpoint.pth.tar")["state_dict"])

# Train
print('Training:')
scaler = torch.cuda.amp.GradScaler() # Don't have cuda
for epoch in range(num_epochs):
    print(f'\tEpoch {epoch}')

    # Train and save snapshot of this epoch's training, in case it crashes while training next one
    funclib.train_UNET(train_loader, model, optimizer, loss_fn, scaler, dm_weight=10, bp_weight=10, device=device) # call model
    state = {"state_dict": model.state_dict(), "optimizer":optimizer.state_dict(),}
    torch.save(state, "../NN_storage/UNET_checkpoint.pth.tar"); print(f'\tSaving checkpoint to ../NN_storage/UNET_checkpoint.pth.tar')

    # check accuracy 
    accuracy, dice_score = funclib.validate(val_loader, model, device)
    print(f"\tGot accuracy {accuracy:.2f} and dice score: {dice_score/len(val_loader)}")
    model.train() # set model back into train mode

# Save model 
torch.save(model.state_dict(), f'../NN_storage/{UNet_name}.pth')
print(f'Saving trained model as {UNet_name}.pth')

# Load it back in and save results on validation data 
model = funclib.MyUNet(in_channels, n_classes)
model.load_state_dict(torch.load(f'../NN_storage/{UNet_name}.pth'))
funclib.save_UNET_results(val_loader, save_dir=f'../{UNet_name}_outputs', model=model)



# UNet_name = 'UNet13'
# n_classes = 4
# multichannel = False
# imdir = "../Data/UNetData_MURaM/images/"
# segdir = "../Data/UNetData_MURaM/seg_images/"
# in_channels = 1
# bp_weight = 5
# dm_weight = 4
# loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1,dm_weight,1,bp_weight])) #funclib.multiclass_MSE_loss() # nn.BCEWithLogitsLoss() #
# load_model = False
# num_epochs = 3 

# # Get the data, applying some transformations (Should pull segs and truths correctly based on their positions in the directories)
# batch_size = 16
# train_ds = funclib.MyDataset(image_dir=f"{imdir}train", mask_dir=f"{segdir}train", n_classes=n_classes, multichannel=multichannel) # multichannel=True, channels=['deltaBinImg'], 
# train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True)
# val_ds = funclib.MyDataset(image_dir=f"{imdir}val", mask_dir=f"{segdir}val", n_classes=n_classes, multichannel=multichannel) # multichannel=True, channels=['deltaBinImg'],
# val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False)
# funclib.check_inputs(train_ds, train_loader)

# # Define model (as an instance of MyNeuralNet), loss function, and optimizer
# model = funclib.MyUNet(in_channels, out_channels=n_classes).to("cpu")
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# if load_model: 
#     model.load_state_dict(torch.load("../NN_storage/UNET_checkpoint.pth.tar")["state_dict"])

# # Train
# print('Training:')
# scaler = torch.cuda.amp.GradScaler() # Don't have cuda
# for epoch in range(num_epochs):
#     print(f'\tEpoch {epoch}')

#     # Train and save snapshot of this epoch's training, in case it crashes while training next one
#     funclib.train_UNET(train_loader, model, optimizer, loss_fn, scaler, dm_weight=dm_weight, bp_weight=bp_weight) # call model
#     state = {"state_dict": model.state_dict(), "optimizer":optimizer.state_dict(),}
#     torch.save(state, "../NN_storage/UNET_checkpoint.pth.tar"); print(f'\tSaving checkpoint to ../NN_storage/UNET_checkpoint.pth.tar')

#     # check accuracy 
#     accuracy, dice_score = funclib.validate_UNET(val_loader, model)
#     print(f"\tGot accuracy {accuracy:.2f} and dice score: {dice_score/len(val_loader)}")
#     model.train() # set model back into train mode

# # Save model 
# torch.save(model.state_dict(), f'../NN_storage/{UNet_name}.pth')
# print(f'Saving trained model as {UNet_name}.pth')

# # Load it back in and save results on validation data 
# model = funclib.MyUNet(in_channels, out_channels=n_classes)
# model.load_state_dict(torch.load(f'../NN_storage/{UNet_name}.pth'))
# funclib.save_UNET_results(val_loader, save_dir=f'../{UNet_name}_outputs', model=model)
