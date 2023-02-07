
import os
import astropy.io.fits as fits
import funclib
import numpy as np

# Get data
dir = os.getcwd()+"/Data/DKIST_gband_series_183653/"
files = os.listdir(dir)
print(files)
for file in files:
    if not file.startswith('seg_'):
        print('Input file '+str(file))
        if os.path.exists(dir+'seg_'+str(file)):
            print('     Segmented data already exists')
        else:
            data = fits.open(dir+file)[1].data # for files in DKIST_gband_series_183653, first HDU is just header
            seg_data = funclib.segment_array(data, resolution=0.016)
            seg_hdu = fits.PrimaryHDU(seg_data)
            raw_hdu = fits.ImageHDU(data)
            hdu = fits.HDUList([seg_hdu, raw_hdu])
            print('     Writing segmented data to'+str(dir)+'seg_'+str(file))
            hdu.writeto(dir+'seg_'+str(file), overwrite=True)