import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import funclib
import cv2
import shutil
import skimage as sk
tc = 'white'

dir = '../Data/All_DKIST/'
badtags = ['31_21', '33_13']
files = [filename for filename in os.listdir(dir) if filename.startswith('VBI') and filename.endswith('_4096') and not any(tag in filename for tag in badtags)] 
# fig, [(ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10), (ax11, ax12), (ax13, ax14), (ax15, ax16), (ax17, ax18), (ax19, ax20), (ax21, ax22), (ax23, ax24), (ax25, ax26), (ax27, ax28), (ax29, ax30), (ax31, ax32), (ax33, ax34), (ax35, ax36)] =  plt.subplots(18, 2, figsize=(25, 200)); axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24, ax25, ax26, ax27, ax28, ax29, ax30, ax31, ax32, ax33, ax34, ax35, ax36]
print(len(files))
print(files)
fig, axs = plt.subplots(26, 2, figsize=(25, 300))
axs = axs.flatten()
j = 0
for i in range(0, 2*len(files)+1, 2):
    print(f'{j}, {files[j]}')
    data = fits.open(dir+files[j])[1].data
    segdata = fits.open(dir+'SEGv2_'+files[j])[0].data
    axs[i].set_title(files[j])
    im = axs[i].imshow(data, origin='lower')
    #plt.colorbar(im, ax=axs[i])
    im = axs[i+1].imshow(segdata, origin='lower')
    #plt.colorbar(im, ax=axs[i+1])
    j += 1

plt.savefig('test0710')