import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import os


##################################
# WNet (unsupervised segmentation)
##################################

class WNet(nn.Module):
    '''
    WNet model for unsupervised image segmentation
    Inputs
     - in_chans: n input channels
     - n_classes: n seg classes
     - out_chans: generally should be same as n input channels (predict encoding same shape as input)
     - kernel_size: size of convolutional kernel
             Note that for a convolutional layer, output size is [(input_size−kernel_size+2*padding)/stride]+1
             So to maintain output size could use
                1) kernel size of 3, stride of 1, padding of 1: [(l_in−3+2*1)/1]+1 = l_in
                2) kernel size of 5, stride of 1, padding of 2: [(l_in-5+2*2)/1]+1 = l_in 
     - padding_model: use 'replicate' (or 'reflect') to aviod framing
    '''

    def __init__(self, in_chans, n_classes, out_chans=1000, dec_depth=4, double_dec=False, dec_ch_mul=64, kernel_size=3, padding_mode=None, reconstruct_mag=False, pretrained_dec=None, activation=False): # 1000 is just a placeholder 
        super(WNet, self).__init__()
        if out_chans==1000:
            out_chans=in_chans
        padding = int((kernel_size - 1)/2)  # so that convolutions maintain size, assuming stride = 1 (e.g. w/ k=3, p=1, and w/ k=5, p=2)
        output_padding = 1 # if kernel_size == 3 else 0 if kernel_size == 5 else 'Error'
        self.UEnc=UEnc(in_chans=in_chans, n_classes=n_classes, kernel_size=kernel_size, padding=padding, output_padding=output_padding, padding_mode=padding_mode, activation=activation)
        self.UDec=UDec(n_classes=n_classes, out_chans=out_chans, depth=dec_depth, double=double_dec, ch_mul=dec_ch_mul, kernel_size=kernel_size, padding=padding,  output_padding=output_padding, padding_mode=padding_mode, reconstruct_mag=reconstruct_mag, activation=activation)
        if str(pretrained_dec) != 'None':
            try:
                pre_trained_dec = torch.load(f'../model_runs_dec/MURaM/{pretrained_dec}/{pretrained_dec}.pth')
                self.UDec.load_state_dict(pre_trained_dec)
            except FileNotFoundError:
                raise FileNotFoundError(f'../model_runs_dec/MURaM/{pretrained_dec}/{pretrained_dec}.pth DNE. Current WD {os.getcwd()}')
            
    def forward(self, x, compute_dec):
        enc = self.UEnc(x) # already softmax
        if compute_dec == False:
            return enc
        else: 
            dec = self.UDec(enc)
            return dec
        
class UEnc(nn.Module):
    '''
    Encoder UNet (img --> seg)
    '''
    def __init__(self, in_chans, n_classes, ch_mul=64, kernel_size=3, padding=1, output_padding=1, padding_mode=None, activation=False):
        super(UEnc, self).__init__()
        
        self.enc1=Block(in_chans, ch_mul, seperable=False, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, activation=activation)
        self.enc2=Block(ch_mul, 2*ch_mul, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, activation=activation)
        self.enc3=Block(2*ch_mul, 4*ch_mul, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, activation=activation)
        self.enc4=Block(4*ch_mul, 8*ch_mul, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, activation=activation)
        
        self.middle=Block(8*ch_mul, 16*ch_mul, padding_mode=padding_mode, activation=activation)
        
        self.up1=nn.ConvTranspose2d(16*ch_mul, 8*ch_mul, kernel_size=kernel_size, stride=2, padding=padding, output_padding=output_padding) # padding_mode only allowed to be 'zeros'
        self.dec1=Block(16*ch_mul, 8*ch_mul, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, activation=activation)
        self.up2=nn.ConvTranspose2d(8*ch_mul, 4*ch_mul, kernel_size=kernel_size, stride=2, padding=padding, output_padding=output_padding)
        self.dec2=Block(8*ch_mul, 4*ch_mul, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, activation=activation)
        self.up3=nn.ConvTranspose2d(4*ch_mul, 2*ch_mul, kernel_size=kernel_size, stride=2, padding=padding, output_padding=output_padding)
        self.dec3=Block(4*ch_mul, 2*ch_mul, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, activation=activation)
        self.up4=nn.ConvTranspose2d(2*ch_mul, ch_mul, kernel_size=kernel_size, stride=2, padding=padding, output_padding=output_padding)
        self.dec4=Block(2*ch_mul, ch_mul, seperable=False, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, activation=activation)
        
        self.final=nn.Conv2d(ch_mul, n_classes, kernel_size=(1, 1), padding_mode=padding_mode) # self.final=nn.Conv2d(ch_mul, squeeze, kernel_size=(1, 1)); self.softmax = nn.Softmax2d()
        
    def forward(self, x): 
        
        enc1=self.enc1(x); #print(x.shape, enc1.shape) # [16, 1, 128, 128] -> [16, 64, 128, 128]  
        enc2=self.enc2(F.max_pool2d(enc1, (2,2))); #print('enc2',enc2.shape) # [16, 64, 128, 128] -> [16, 128, 64, 64]         
        enc3=self.enc3(F.max_pool2d(enc2, (2,2))); #print('enc3',enc3.shape) # [16, 128, 64, 64] -> [16, 256, 32, 32]         
        enc4=self.enc4(F.max_pool2d(enc3, (2,2))); #print('enc4',enc4.shape)# [16, 256, 32, 32] -> [16, 512, 16, 16] 
        
        middle=self.middle(F.max_pool2d(enc4, (2,2))); #print('middle',middle.shape) # [16, 512, 16, 16] -> [16, 1024, 8, 8]
        
        up1=torch.cat([enc4, self.up1(middle)], 1); #print('up1',up1.shape) # [16, 512, 16, 16] + self.up1(middle):[16, 512, 16, 16] 
        dec1=self.dec1(up1);# print('dec1',dec1.shape)
        up2=torch.cat([enc3, self.up2(dec1)], 1); #print('up2',up2.shape)
        dec2=self.dec2(up2); #print('dec2',dec2.shape)
        up3=torch.cat([enc2, self.up3(dec2)], 1); #print('up3',up3.shape)
        dec3=self.dec3(up3); #print('dec3',dec3.shape)
        up4=torch.cat([enc1, self.up4(dec3)], 1); #print('up4',up4.shape)
        dec4=self.dec4(up4); #print('dec4',dec4.shape)
        
        final=self.final(dec4)
        
        return F.softmax(final, 1) 

class UDec_old(nn.Module):
    '''
    Decoder UNet (seg --> img)
    '''
    def __init__(self, n_classes, out_chans, ch_mul=64, kernel_size=3, padding=1, output_padding=1, padding_mode=None, reconstruct_mag=False, depth=4, double=False, activation=False):
                
        #  depth, double, and activation NOT USED - just added temporarily for compatability 
        super(UDec_old, self).__init__()
        
        if eval(str(reconstruct_mag)): 
            print(f'reconstruct_mag is true')
            out_chans = out_chans + 1
        
        self.enc1=Block(n_classes, ch_mul, seperable=False, padding_mode=padding_mode)
        self.enc2=Block(ch_mul, 2*ch_mul, padding_mode=padding_mode)
        self.enc3=Block(2*ch_mul, 4*ch_mul, padding_mode=padding_mode)
        self.enc4=Block(4*ch_mul, 8*ch_mul, padding_mode=padding_mode)
        
        self.middle=Block(8*ch_mul, 16*ch_mul, padding_mode=padding_mode)
        
        self.up1=nn.ConvTranspose2d(16*ch_mul, 8*ch_mul, kernel_size=kernel_size, stride=2, padding=padding, output_padding=output_padding) # padding_mode only allowed to be 'zeros'
        self.dec1=Block(16*ch_mul, 8*ch_mul, padding_mode=padding_mode)
        self.up2=nn.ConvTranspose2d(8*ch_mul, 4*ch_mul, kernel_size=kernel_size, stride=2, padding=padding, output_padding=output_padding)
        self.dec2=Block(8*ch_mul, 4*ch_mul, padding_mode=padding_mode)
        self.up3=nn.ConvTranspose2d(4*ch_mul, 2*ch_mul, kernel_size=kernel_size, stride=2, padding=padding, output_padding=output_padding)
        self.dec3=Block(4*ch_mul, 2*ch_mul, padding_mode=padding_mode)
        self.up4=nn.ConvTranspose2d(2*ch_mul, ch_mul, kernel_size=kernel_size, stride=2, padding=padding, output_padding=output_padding)
        self.dec4=Block(2*ch_mul, ch_mul, seperable=False, padding_mode=padding_mode)
        
        self.final=nn.Conv2d(ch_mul, out_chans, kernel_size=(1, 1), padding_mode=padding_mode)
        
    def forward(self, x):
        
        enc1 = self.enc1(x);# print(f'down 0 {enc1.shape}')
        enc2 = self.enc2(F.max_pool2d(enc1, (2,2))); #print(f'down 1 {enc2.shape}')
        enc3 = self.enc3(F.max_pool2d(enc2, (2,2))); #print(f'down 2 {enc3.shape}')
        enc4 = self.enc4(F.max_pool2d(enc3, (2,2))); #print(f'down 3 {enc4.shape}')
        
        middle = self.middle(F.max_pool2d(enc4, (2,2))); #print(f'middle {middle.shape}')
        
        up1 = torch.cat([enc4, self.up1(middle)], 1)
        dec1 = self.dec1(up1); #print(f'dec0(up0) {dec1.shape}')
        up2 = torch.cat([enc3, self.up2(dec1)], 1)
        dec2 = self.dec2(up2); #print(f'dec1(up1) {dec2.shape}')
        up3 = torch.cat([enc2, self.up3(dec2)], 1)
        dec3 =self.dec3(up3); #print(f'dec2(up2) {dec3.shape}')
        up4 = torch.cat([enc1, self.up4(dec3)], 1)
        dec4 = self.dec4(up4); #print(f'dec3(up3) {dec4.shape}')
        
        final=self.final(dec4)
        
        return final

    
class UDec(nn.Module):
    '''
    NOTE: this *should* be the same as the above (using depth=4 and double=False), but for some reason works way better...
    Decoder UNet (seg --> img)
    '''
    def __init__(self, n_classes, out_chans=3, depth=4, double=False, ch_mul=64, kernel_size=3, padding=1, output_padding=1, padding_mode=None, reconstruct_mag=False, activation=False):
        super(UDec, self).__init__()
        
        
        if eval(str(reconstruct_mag)): 
            print(f'reconstruct_mag is true')
            ch_out = ch_out + 1

        self.double = double
        self.downs = nn.ModuleList()
        self.downs.append(Block(n_classes, ch_mul, seperable=False, padding_mode=padding_mode, activation=activation))
        ch_in = ch_mul
        for i in range(depth-1): # e.g. for depth=4 will end up with 4 layers, since first added above and this goes 0, 1, 2
            ch_out = 2*ch_in
            self.downs.append(Block(ch_in, ch_out, padding_mode=padding_mode, activation=activation))
            if double: self.downs.append(Block(ch_out, ch_out, padding_mode=padding_mode, activation=activation))
            ch_in = ch_out

        self.middle=Block(ch_out, ch_out*2, padding_mode=padding_mode, activation=activation) # ch_out will be ch_out from last down
        
        self.ups = nn.ModuleList() 
        self.decs = nn.ModuleList()
        ch_in = ch_out*2
        for i in range(depth):
            ch_out = int(ch_in/2)
            seperable = False if i == depth-1 else True
            self.ups.append(nn.ConvTranspose2d(ch_in, ch_out, kernel_size=kernel_size, stride=2, padding=padding, output_padding=output_padding))
            self.decs.append(Block(ch_in, ch_out, seperable=seperable, padding_mode=padding_mode, activation=activation))
            if double and i!=depth-1: # DONT DOUBLE LAST ONE TO MIRROR UPS
                self.ups.append(nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, padding=1, padding_mode=padding_mode))  # ADDING CONV INSTEAD OF CONVTRANSPOSE AS "FILLER" TO PRESERVE PIX DIMS... THAT SHOULD BE OK RIGHT?
                self.decs.append(Block(2*ch_out, ch_out, seperable=seperable, padding_mode=padding_mode, activation=activation)) # 2*CH_OUT TO ACCOUNT FOR CONCAT
            ch_in = ch_out
        self.final = nn.Conv2d(ch_mul, out_chans, kernel_size=(1, 1), padding_mode=padding_mode)
        
    def forward(self, X):
        
        X = self.downs[0](X)#; print(f'down{0}: {X.shape}')
        skips = [X] # add first down to skips             
        for i in range(len(self.downs)-1):
            if self.double and (i+1)%2 == 0: # if we've doubled up on layers, only pool before 1st (e.g. after 0th), 3rd, 5th, etc, so that we dont run our of pixels
                X = self.downs[i+1](X)#; print(f'down {i+1}: {X.shape}')
            else:
                X = self.downs[i+1](F.max_pool2d(X, (2,2)))#; print(f'down(pool) {i+1}: {X.shape}')
            skips.append(X) 
                            
        X = self.middle(F.max_pool2d(X, (2,2)))#; print(f'middle {X.shape}')
        
        for i in range(len(self.ups)):
            skip = skips[-(i+1)]
            #print(f'  skip{-(i+1)}: {skip.shape}')
            #print(f'  up{i}: {self.ups[i](X).shape}');
            X = torch.cat([skip, self.ups[i](X)],1)#; print(f'  cat{i}: {X.shape}')
            X = self.decs[i](X)#; print(f'  dec{i}: {X.shape}')
                           
        final = self.final(X)
        
        return final
    
    
class Block(nn.Module):
    '''
    Container for multiple layers (for convenience)
        - ADD KERNEL SIZE!!!??
    '''
    def __init__(self, in_filters, out_filters, seperable=True, kernel_size=3, padding=1, padding_mode=None, activation=False):
        super(Block, self).__init__()
        
        if seperable:
            self.spatial1=nn.Conv2d(in_filters, in_filters, kernel_size=kernel_size, groups=in_filters, padding=padding, padding_mode=padding_mode)
            self.depth1=nn.Conv2d(in_filters, out_filters, kernel_size=1, padding_mode=padding_mode)
            self.conv1=lambda x: self.depth1(self.spatial1(x))
            self.spatial2=nn.Conv2d(out_filters, out_filters, kernel_size=kernel_size, padding=padding, groups=out_filters, padding_mode=padding_mode)
            self.depth2=nn.Conv2d(out_filters, out_filters, kernel_size=1, padding_mode=padding_mode)
            self.conv2=lambda x: self.depth2(self.spatial2(x))
            
        else:
            self.conv1=nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode)
            self.conv2=nn.Conv2d(out_filters, out_filters, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode)
        
        self.activation = activation
        self.act = nn.LeakyReLU(); self.dropout = nn.Dropout(0.65)  # from reproduction
        self.batchnorm1=nn.BatchNorm2d(out_filters)
        #self.relu2 = nn.ReLU(); self.dropout2 = nn.Dropout(0.65)  # from reproduction
        self.batchnorm2=nn.BatchNorm2d(out_filters) 

    def forward(self, x):

        x=self.batchnorm1(self.conv1(x)).clamp(0) 
        if self.activation: 
            x = self.dropout(self.act(x)) # x = self.relu1(x); x = self.dropout1(x)  # from reproduction
        x=self.batchnorm2(self.conv2(x)).clamp(0)
        if self.activation:
            x = self.dropout(self.act(x)) # x = self.relu2(x); x = self.dropout2(x)  # from reproduction

        return x

######################################
# UNet (supervised image segmentation)
######################################

class MyUNet(nn.Module):
    '''
    UNet class
    '''
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512],):
        
        super(MyUNet, self).__init__() 
        self.ups = nn.ModuleList() 
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature  # after each convolution we set (next) in_channel to (previous) out_channels  
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2,))
            self.ups.append(DoubleConv(feature*2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    

    def forward(self, x): 

        skip_connections = []
        for down in self.downs:
            x = down(x) 
            skip_connections.append(x) 
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # reverse 
        for idx in range(0, len(self.ups), 2): # step of 2 becasue add conv step
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=None)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    

class DoubleConv(nn.Module):
    '''
    Containor for conv sets (for convenience)
    '''
    def __init__(self, in_channels, out_channels):

        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)



######################################
# "Simple" CNNS (supervised image prediction)
######################################

class magNet0(nn.Module):
    '''
    *Very* simple CNN (constant N_pix dimension)
    '''
    def __init__(self, in_channels, hidden_channels=[16, 64], k_size=3):
        super(magNet0, self).__init__()
        self.act = nn.ReLU()
        self.convs = nn.ModuleList()
        in_chans = in_channels
        for hc in hidden_channels:
            self.convs.append(nn.Conv2d(in_channels=in_chans, out_channels=hc, kernel_size=3, stride=1, padding=1))
            self.convs.append(self.act)
            in_chans = hc
        self.convs.append(nn.Conv2d(in_channels=hidden_channels[-1], out_channels=1, kernel_size=3, stride=1, padding=1))
        self.convs.append(self.act)
        
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)

        return x
    
    
    
class DeepVel(nn.Module):
    '''
    CNN based on DeepVel model (which is based on ResNet)
    '''
    def __init__(self, in_channels, hidden_channels=64, k_size=3, N_blocks=20):
        super(DeepVel, self).__init__()
        padding = int((k_size - 1)/2)  # so that convolutions maintain size, assuming stride = 1 (e.g. w/ k=3, p=1, and w/ k=5, p=2)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=k_size, stride=1, padding=padding)
        self.residual_blocks = nn.ModuleList()
        for i in range(N_blocks):
            self.residual_blocks.append(ResBlock(hidden_channels)) # they keep same hidden_channels (64) throughout
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=k_size, stride=1, padding=padding)
        self.norm = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        
    def forward(self, X):
        x = self.act(self.conv1(X))
        skip = torch.clone(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.norm(self.conv2(x))
        x = x + skip 
        out = self.conv3(x)
        
        return out
    
    
class ResBlock(nn.Module):
    def __init__(self, hidden_channels=64): # they keep hidden channels constant at 64, so no in_channels and out_channels
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm2d(hidden_channels)
        
    def forward(self, x):
        skip = torch.clone(x)
        x = self.norm(self.conv(x))
        x = self.act(x)
        x = self.norm(self.conv(x))
        out =  x + skip
        
        return out
        
# class regVGG(nn.Module):
#     '''
#     CNN based on VGG16 model, modified to predict image instead of segmentation (pixel regression instead of full image classification)
#     '''
#     def __init__(self, in_channels):
#         super(regVGG, self).__init__() 
#         self.convs = nn.Sequential(
#             DoubleConv(in_channels, 64), 
#             DoubleConv(64, 128), 
#             TripleConv(128, 256), 
#             TripleConv(256, 512), 
#             TripleConv(512, 512))
#         self.linear1 = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(16*16*512, 4096), #nn.Linear(7*7*512, 4096), # 4096
#             nn.ReLU())
#         self.linear2 = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(4096, 4096), # 4096
#             nn.ReLU())
#         self.linear3 = nn.Sequential(
#             nn.Linear(4096, 1)) # 4096
        
#     def forward(self, X):
#         out = self.convs(X) # [16, 1, 227, 227] -> [16, 512, 16, 16]    ## [16, 512, 7, 7]
#         out = out.reshape(out.size(0), -1)# [16, 512, 16, 16] -> [16, 131072]     ## [16, 512, 7, 7] -> [16, 25088] 
#         out = self.linear1(out) # [16, 131072] -> [16, 2048]    ## [16, 25088] -> [16, 2048]  
#         out = self.linear2(out) # [16, 2048] -> [16, 2048]
#         out = self.linear3(out) # [16, 2048] -> [16, 3]
#         return out
    
# class DoubleConv(nn.Module):
#     def __init__(self, in_chans, out_chans):
#         super(DoubleConv, self).__init__()
#         self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1)
#         self.batchnorm = nn.BatchNorm2d(out_chans)
#         self.relu  = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

#     def forward(self, X):
#         X = self.relu(self.batchnorm(self.conv1(X)))
#         X = self.relu(self.batchnorm(self.conv2(X)))
#         X = self.maxpool(X)
#         return X 

# class TripleConv(nn.Module):
#     def __init__(self, in_chans, out_chans):
#         super(TripleConv, self).__init__()
#         self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1)
#         self.batchnorm = nn.BatchNorm2d(out_chans)
#         self.relu  = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

#     def forward(self, X):
#         X = self.relu(self.batchnorm(self.conv1(X)))
#         X = self.relu(self.batchnorm(self.conv2(X)))
#         X = self.relu(self.batchnorm(self.conv3(X)))
#         X = self.maxpool(X)
#         return X
    
    
    
######################################
# magUNet (supervised image prediction) - BUT OVERKILL, SEE ABOVE
######################################

class magUNet(nn.Module):
    '''
    Based on UNet class, but performs image prediction not segmentation
    '''
    def __init__(self, in_channels=1, features=[64, 128, 256, 512],):
        
        super(magUNet, self).__init__() 
        self.ups = nn.ModuleList() 
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature  # after each convolution we set (next) in_channel to (previous) out_channels  
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2,))
            self.ups.append(DoubleConv(feature*2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], 1, kernel_size=1) # predict one out channel (mag image)
    

    def forward(self, x): 

        skip_connections = []
        for down in self.downs:
            x = down(x) 
            skip_connections.append(x) 
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # reverse 
        for idx in range(0, len(self.ups), 2): # step of 2 because add conv step
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=None)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    

class DoubleConv(nn.Module):
    '''
    Containor for conv sets (for convenience)
    '''
    def __init__(self, in_channels, out_channels):

        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


# class Block(nn.Module):
#     '''
#     Containor for multiple layers (for convenience)
#     '''
#     def __init__(self, in_filters, out_filters, seperable=True, padding=1, padding_mode=None):
#         super(Block, self).__init__()
        
#         if seperable:
#             self.spatial1=nn.Conv2d(in_filters, in_filters, kernel_size=3, groups=in_filters, padding=padding, padding_mode=padding_mode)
#             self.depth1=nn.Conv2d(in_filters, out_filters, kernel_size=1, padding_mode=padding_mode)
#             self.conv1=lambda x: self.depth1(self.spatial1(x))
#             self.spatial2=nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=padding, groups=out_filters, padding_mode=padding_mode)
#             self.depth2=nn.Conv2d(out_filters, out_filters, kernel_size=1, padding_mode=padding_mode)
#             self.conv2=lambda x: self.depth2(self.spatial2(x))
            
#         else:
#             self.conv1=nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=padding, padding_mode=padding_mode)
#             self.conv2=nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=padding, padding_mode=padding_mode)

#         # self.relu1 = nn.ReLU(); self.dropout1 = nn.Dropout(0.65)  # from reproduction
#         self.batchnorm1=nn.BatchNorm2d(out_filters)
#         # self.relu2 = nn.ReLU(); self.dropout2 = nn.Dropout(0.65)  # from reproduction
#         self.batchnorm2=nn.BatchNorm2d(out_filters) 

#     def forward(self, x):

#         x=self.batchnorm1(self.conv1(x)).clamp(0) 
#         # x = self.relu1(x); x = self.dropout1(x)  # from reproduction
#         x=self.batchnorm2(self.conv2(x)).clamp(0)
#         # x = self.relu2(x); x = self.dropout2(x)  # from reproduction

#         return x


