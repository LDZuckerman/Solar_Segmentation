
import os, shutil
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Script to prepare all NN datasets from given set of photometric images
# Assumes "truth" algorithmic segs already created via run_segment_aglorithm.py
# Can athen prepare analogous datasets of mag, vel, etc images, with add_set.py
###############################################################################

'''
Params
'''
name = 'MURaM'
dpath = '../../Data'
raw_imgdir = f'{dpath}/{name}' # e.g. ../../Data/MURaM
raw_magdir = f'{dpath}/{name}_mag' # e.g. ../../Data/MURaM_mag
raw_veldir = f'{dpath}/{name}_vel' # e.g. ../../Data/MURaM_vel
UNetData_dir =f'{dpath}/UNetData_{name}' # e.g. ../../Data/UNetData_MURaM
norm_type = 'full' # normalize by full series 
bad = []
startswith = 'I'
endswith = ''
n = 3 # sqrt of num pieces to break each image into
pos = 0

norm_tag = 'full' if norm_type == 'full' else ''
subsecdir = f'{UNetData_dir}/{norm_tag}norm_images' # e.g. ../../Data/UNetData_MURaM/fullnorm_images
segsubsecdir = f'{UNetData_dir}/seg_images'  # e.g. ../../Data/UNetData_MURaM/seg_images
raw_imgfiles = [file for file in os.listdir(raw_imgdir) if file.startswith(startswith) and file.endswith(endswith) and not any(tag in file for tag in bad)]
trnimgdir = f"{subsecdir}/train"; tstimgdir = f"{subsecdir}/val"; trnsegdir = f"{segsubsecdir}/train"; tstsegdir = f"{segsubsecdir}/val"  
    
# '''
# Find full series min and max
# '''
# if norm_type == 'full' and full_min == "calculate":
#     mins = []
#     maxs = []
#     for imgfile in raw_imgfiles:
#         img = fits.open(f"{raw_imgdir}/{imgfile}")[pos].data
#         mins.append(np.nanmin(img))
#         maxs.append(np.nanmax(img))
#     full_min = np.nanmin(mins)
#     full_max = np.nanmax(maxs)
#     print(f'Min and max of all {len(raw_imgfiles)} OG files: {full_min, full_max}', flush=True)

# '''
# Make normalized versions, save in raw dir
# '''
# if len([f for f in os.listdir(raw_imgdir) if f'{norm_tag}Norm' in f]) == 0:
#     print(f'Saving full-series normalized versions of {len(raw_imgfiles)} OG files', flush=True)
#     count = 0
#     for imgfile in raw_imgfiles:
#         img = fits.open(f"{raw_imgdir}/{imgfile}")[pos].data
#         header = fits.open(f"{raw_imgdir}/{imgfile}")[0].header
#         if norm_type == 'individual':
#             img_norm = (img - np.nanmin(img))/(np.nanmax(img) - np.nanmin(img))
#         else:
#             img_norm = (img - full_min)/(full_max - full_min)
#         if np.max(img_norm) > 1:
#             raise ValueError('Data has not been normalized')
#         hdu = fits.PrimaryHDU(img_norm)
#         hdu.writeto(f"{raw_imgdir}/{norm_tag}Norm_{imgfile}", overwrite=True)
#         count += 1
#     print(f'Saved {count} files (now {len([file for file in os.listdir(raw_imgdir) if file.startswith(f"{norm_tag}Norm")])} normed files)', flush=True)

# '''
# Make subsections of normed images and place in subsection dir (e.g. UNetData_MURaM/fullNorm_images/)
# ''' 
# if not os.path.exists(subsecdir): 
#     print(f'Creating {subsecdir}', flush=True)
#     os.mkdir(subsecdir) 
# if not os.path.exists(segsubsecdir): 
#     print(f'Creating {segsubsecdir}', flush=True)
#     os.mkdir(segsubsecdir) 
# normfiles = [file for file in os.listdir(raw_imgdir) if file.startswith(f'{norm_tag}Norm')] # files = [fil for file in os.listdir(OG_dir) if file.startswith('Norm_VBI') and file.endswith('_4096') and not any(tag in file for tag in ['31_21', '33_13', '04_29', '17_20', '24_26'])]   
# pad = int(np.shape(fits.open(f'{raw_imgdir}/{raw_imgfiles[0]}')[0].data)[0]/200)
# print(f'Saving subsections for {len(normfiles)} files', flush=True)
# for file in normfiles:
#     if 'MURaM' in UNetData_dir: tag = file[file.find('out')+4: file.find('.fits')]
#     else: tag = file[file.find('VBI'):-5]
#     if os.path.exists(f'{subsecdir}/{tag}__{n**2-1}.npy'): # last .npy created from normfile
#         print(f'\tNormed file {file} (tag {tag}) --> subsections already saved', flush=True)
#     elif os.path.exists(f'{subsecdir}/train/{tag}__{n**2-1}.npy') or os.path.exists(f'{subsecdir}/val/{tag}__{n**2-1}.npy'): # last .npy created from normfile
#         print(f'\tNormed file {file} (tag {tag}) --> subsections already saved and split into train/test', flush=True)
#     else:
#         #print(f'{subsecdir}/{tag}__{n**2-1}.npy DNE');a=b
#         print(f'\tNormed file {file} (tag {tag}) --> saving subsections', flush=True)
#         seg = np.squeeze(fits.open(f'{raw_imgdir}/SEGv2_{file.replace("fullNorm_","")}')[0].data)
#         seg = seg[pad:-pad, pad:-pad] # Remove zero padding at edges of seg file
#         data = fits.open(f'{raw_imgdir}/{file}')[0].data 
#         data = data[pad:-pad, pad:-pad] # Remove zero padding at edges of seg file
#         N = np.min(np.shape(data)) 
#         length = int(N/n) 
#         num = 0
#         for i in range(n):
#             for j in range(n):
#                 name = tag+'__'+str(num); num += 1
#                 x1 = length * i
#                 x2 = length * (i + 1)
#                 y1 = length * j
#                 y2 = length * (j + 1)
#                 savedata = data[x1:x2, y1:y2]
#                 saveseg = seg[x1:x2, y1:y2]
#                 np.save(f'{subsecdir}/{name}', savedata)
#                 np.save(f'{segsubsecdir}/SEG_{name}', saveseg)
#     if not os.path.exists(f'{subsecdir}/{tag}__{n**2-1}.npy'):
#         raise ValueError(f'\tSomething went wrong. Didnt save all subsections of {tag}')
# npyfiles = [file for file in os.listdir(subsecdir) if '.npy' in file] # all files that aren't folders (train and test)
# print(f'Saved {len(npyfiles)} total subsection npy files, each with shape {np.shape(np.load(f"{subsecdir}/{npyfiles[0]}"))}', flush=True)
# print(f'Make sure this matches {n**2 * len(normfiles)}', flush=True) #{len(os.listdir(f"{UNetData_dir}/images/train") + os.listdir(f"{UNetData_dir}/images/val"))} in old subsec dir') # no idea why there are a few more npy files for all_dkist... but shouldnt matter becasue just won't be moved to train and val folders right?
# # Check none are missing
# subsecs = os.listdir(subsecdir)
# for i in range(24): # og imgs go from I_out.000000.fits.gz to I_out.018000.fits.gz
#     tag = "{:06d}".format(i*10)
#     if len([f for f in subsecs if tag in f]) != n**2:
#         raise ValueError(f'File {tag} has only {len([f for f in subsecs if tag in f])} subsections saved, not {n**2}')
    
    
# '''
# Split into train and test 
# '''
# for d in [trnimgdir, tstimgdir, trnsegdir, tstsegdir]:
#     if os.path.exists(d)==False: 
#         print(f'Creating {d}', flush=True)
#         os.mkdir(d)
# imgfiles = [file for file in os.listdir(subsecdir) if '.npy' in file]
# print(f'Splitting the {len(imgfiles)} subsection files into train and test', flush=True)
# for imgfile in imgfiles:
#     label = imgfile[:-4]
#     segfile = [filename for filename in os.listdir(segsubsecdir) if label in filename][0]
#     folder = 'train' if np.random.rand(1)[0] < 0.7 else 'val'
#     shutil.move(f"{subsecdir}/{imgfile}", f"{subsecdir}/{folder}/{imgfile}")
#     shutil.move(f"{segsubsecdir}/{segfile}", f"{segsubsecdir}/{folder}/{segfile}")
# print(f'Moved {len(os.listdir(trnimgdir))} to train and  {len(os.listdir(tstimgdir))} to test.', flush=True)
# if len(os.listdir(trnimgdir))+len(os.listdir(tstimgdir)) != len(imgfiles):
#     raise ValueError(f'Number of moved subsec files does not equal total number of subsec files')        
    

'''
Create TS UNetData subsec set
    - Make sure to keep correspondence with UNetData_MURaM_TSeries*/seg_images (tags must indicate the same image, and TTS must be the same)
'''

sizes = [21] # 40 should have 41 imgs per set, e.g. 20 imgs (40 sec, for MURaM) on either side of target
for size in sizes:
    UNetTSData_dir = f"../../Data/UNetData_MURaM_TSeries{(size-1)*2}"  
    TSsubsecdir = f"{UNetTSData_dir}/{norm_tag}norm_images"; TStrnimgdir = f"{TSsubsecdir}/train"; TStstimgdir = f"{TSsubsecdir}/val"; TSsegsubsecdir = f"{UNetTSData_dir}/seg_images"; TStrnsegdir = f"{TSsegsubsecdir}/train"; TStstsegdir = f"{TSsegsubsecdir}/val"
    for d in [UNetTSData_dir, TSsubsecdir, TStrnimgdir, TStstimgdir, TSsegsubsecdir, TStrnsegdir, TStstsegdir]:
        if os.path.exists(d)==False: 
            print(f'Creating {d}', flush=True)
            os.mkdir(d)

        # Temporarily copy all (train + val) normed npy img and seg subsections into UNetData_*_TSeries*/{tag}norm_images/ [include parent dir in case some not in train or val] 
        all_npy = [f'{tstimgdir}/{f}' for f in os.listdir(tstimgdir)] + [f'{trnimgdir}/{f}' for f in os.listdir(trnimgdir)] + [f'{tstsegdir}/{f}' for f in os.listdir(tstsegdir)] + [f'{trnsegdir}/{f}' for f in os.listdir(trnsegdir)]     #[f for f in os.listdir(subsecdir) if '.npy' in f] #[f for f in os.listdir(f"{unetdata_dir}/{tag}norm_images") if '.npy' in f] + [f"../../Data/UNetData_MURaM/{tag}norm_images/train/{f}" for f in os.listdir(f"../../Data/UNetData_MURaM/{tag}norm_images/train")] + [f"../../Data/UNetData_MURaM/{tag}norm_images/val/{f}" for f in os.listdir(f"../../Data/UNetData_MURaM/{tag}norm_images/val")]
        print(f'Temporarily copying all {len(all_npy)} imgs from {tstimgdir} and {trnimgdir} into {UNetTSData_dir}', flush=True)
        for d in all_npy:
            file = d[d.rfind('/')+1:]
            if not os.path.exists(f"{UNetTSData_dir}/{file}"):
                shutil.copy(d, f"{UNetTSData_dir}/{file}") #     if os.path.exists(f"{parent}/norm_images/{file}") == False:

    # Split imgs into train and test based on seg location, keeping sections together as cubes
    print(f'Using seg locations in {segsubsecdir} to assign train and val', flush=True)
    subsecfiles = [file for file in os.listdir(UNetTSData_dir) if '.npy' in file and 'SEG' not in file]; 
    subsecsegfiles = [file for file in os.listdir(UNetTSData_dir) if '.npy' in file and 'SEG' in file]
    shape = np.load(f"../../Data/UNetData_MURaM/images/train/000130__3.npy").shape # example image 
    for region in range(n**2): # for each spatial region
        files_region = np.sort([file for file in subsecfiles if f"__{region}" in file])
        segfiles_region = np.sort([file for file in subsecsegfiles if f"__{region}" in file])
        print(f'Collected {len(files_region)} imgs that correspond to region {region}. Will break into {int(len(files_region)/size)} total sets.', flush=True)
        start_step = 0
        while start_step <= 1801-size: # for each set of size imgs (there are 1801 total)
            segname = segfiles_region[start_step + int((size-1)/2)]
            seg = np.load(f"{UNetTSData_dir}/{segname}") # seg for the target image
            files_set = files_region[start_step:start_step+size]
            imgset = np.zeros((size, shape[0], shape[1]))
            for i in range(size-1):
                print(f'   loading {UNetTSData_dir}/{files_set[i]}', flush=True)
                imgset[i,:,:] = np.load(f"{UNetTSData_dir}/{files_set[i]}") # BUT HOW DID I KNOW THIS WOULD BE IN ORDER WITHOUT SORTING IVE NOW APPLIED ABOVE TO IMGFILES_REGION??
            imgsetname = f"{files_set[0][0:-7]}to{files_set[-1][0:-7]}_{region}"
            segsetname = f"SEG_{imgsetname}.npy" # not really a set, just label with set name
            print(f'imgsetname {imgsetname}, segname {segsetname}', flush=True)
            if segname in os.listdir(f"{segsubsecdir}/train/"): 
                folder = 'train'
            elif segname in os.listdir(f"{segsubsecdir}/val/"):
                folder = 'val'
            else:
                raise ValueError(f'Corresponding segmented image {segname} not found')
            np.save(f"{TSsubsecdir}/{folder}/{imgsetname}", imgset)
            np.save(f"{TSsegsubsecdir}/{folder}/{segname}", seg)
            start_step += size   

    # Remove temporarily copy all normed npy subsections
    for file in [file for file in os.listdir(UNetTSData_dir) if '.npy' in file]:
        os.remove(f"{UNetTSData_dir}/{file}")
   

