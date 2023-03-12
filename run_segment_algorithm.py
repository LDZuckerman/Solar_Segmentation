
import os
import astropy.io.fits as fits
import funclib
import numpy as np

# # Run on 'new' DKIST data - cut down to 1/9 the size
# dir = os.getcwd()+"/Data/DKIST_gband_series_183653/"
# files = os.listdir(dir)
# print(files)
# for file in files:
#     if not (file.startswith('seg_') or file.startswith('SEG_')):
#         print('Input file '+str(file))
#         if os.path.exists(dir+'SEG_'+str(file)):
#             print('     Segmented data already exists')
#         if '53_20' in file: # still not sure what the error is for this file
#             print('     skipping file ')
#         else:
#             data = fits.open(dir+file)[1].data # for files in DKIST_gband_series_183653, first HDU is just header
#             data = data[0:int(np.shape(data)[0]/3), 0:int(np.shape(data)[1]/3)] # cut data by a factor of 9
#             header = fits.open(dir+file)[1].header
#             seg_data = funclib.segment_array(data, resolution=header['cdelt1'], mark_dim_centers=True), #resolution=0.016)
#             seg_hdu = fits.PrimaryHDU(seg_data)
#             raw_hdu = fits.ImageHDU(data)
#             hdu = fits.HDUList([seg_hdu, raw_hdu])
#             print('     Writing segmented data to SEG_'+str(file))
#             hdu.writeto(dir+'SEG_'+str(file), overwrite=True) # hdu.writeto(str(dir)+'SEG_'+str(file), overwrite=True)

# # Run on 'old' DKIST data
# file = os.getcwd()+"/Data/DKIST_example.fits"
# data = fits.open(file)[0].data 
# header = fits.open(file)[0].header
# seg_data = funclib.segment_array(data, resolution=0.016, mark_dim_centers=True, mark_BP=True), #resolution=0.016)
# seg_hdu = fits.PrimaryHDU(seg_data)
# raw_hdu = fits.ImageHDU(data)
# hdu = fits.HDUList([seg_hdu, raw_hdu])
# hdu.writeto('Data/DKIST_solarseg_output1.fits', overwrite=True)

# # Run on 'old' DKIST data, but without BPs and DMs
# file = os.getcwd()+"/Data/DKIST_example.fits"
# data = fits.open(file)[0].data 
# header = fits.open(file)[0].header
# seg_data = funclib.segment_array(data, resolution=0.016, mark_dim_centers=False, mark_BP=False), #resolution=0.016)
# seg_hdu = fits.PrimaryHDU(seg_data)
# raw_hdu = fits.ImageHDU(data)
# hdu = fits.HDUList([seg_hdu, raw_hdu])
# hdu.writeto('Data/DKIST_solarseg_output_noBPDMs.fits', overwrite=True)

# Run on new and OG DKIST data, creating lots of small files
dir = os.getcwd()+"/Data/DKIST_gband_series_183653/"
imgoutdir = os.getcwd()+"/Data/UNetData/images/"
segoutdir = os.getcwd()+"/Data/UNetData/seg_images/"
files = os.listdir(dir)
print(files)
for file in files:
    if not (file.startswith('seg_') or file.startswith('SEG_')):
        print('Input file '+ file)
        label =  file[file.find('03T')+6:file.find('_00')-4]
        if '53_20' in file: print('     Skipping file ') # still not sure what the error is for this file
        else:
            # Segment the full file
            data = fits.open(dir+file)[1].data # for files in DKIST_gband_series_183653, first HDU is just header
            header = fits.open(dir+file)[1].header
            seg_data = funclib.segment_array(data, resolution=header['cdelt1'], mark_dim_centers=True), #resolution=0.016)
            # Break it it into 10 (?) sections, save each as a different image (just save as .npy)
            N = np.min(np.shape(data)) 
            n = 5 # sqrt of num pieces to break into
            len = int(N/n) # len of each piece (area = len**2)
            num = 0
            for i in range(n):
                for j in range(n):
                    label = label + str(num); num += 1
                    x1 = len * i
                    x2 = len * (i + 1)
                    y1 = len * j
                    y2 = len * (j + 1)
                    saveimage =  data[x1:x2, y1:y2]
                    saveseg = seg_data[x1:x2, y1:y2]
                    # print(str(num)+' -  '+str(x1)+':'+str(x2)+', '+str(y1)+':'+str(y2))
                    print('     Saving data piece to '+outdir+'_'+label)
                    print('     Saving seg piece to '+segoutdir+'_'+label)
                    np.save(outdir+'_'+label, saveimage)
                    np.save(segoutdir+'_'+label, saveseg)