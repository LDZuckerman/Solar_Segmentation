import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F


##################################
# WNet (unsupervised segmentation)
##################################

class MyWNet(nn.Module):
    '''
    WNet class
    '''

    def __init__(self, squeeze, ch_mul=64, in_chans=3, out_chans=1000, padding_mode=None): # 1000 is just a placeholder 
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
        
class UEnc(nn.Module):
    '''
    Encoder UNet (img --> seg)
    '''
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
        
        return final 

class UDec(nn.Module):
    '''
    Decoder UNet (seg --> img)
    '''
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


##################################
# UNet (supervised segmentation)
##################################

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
    Containor for conve sets (for convenience)
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


class Block(nn.Module):
    '''
    Containor for multiple layers (for convenience)
    '''
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
    


