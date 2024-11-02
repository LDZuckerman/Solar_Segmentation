
import os, shutil
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sndi

'''
Params
'''
raw_imgdir = '../../Data/MURaM'
UNetData_dir = '../../Data/UNetData_MURaM'
norm_type = 'full'
full_min = 13085600000
full_max = 61988004000
bad = []
startswith = 'I'
endswith = ''
n = 3 # sqrt of num pieces to break each image into
pos = 0


'''
Find full series min and max
'''
raw_imgfiles = [file for file in os.listdir(raw_imgdir) if file.startswith(startswith) and file.endswith(endswith) and not any(tag in file for tag in bad)]
if norm_type == 'full' and full_min == "calculate":
    mins = []
    maxs = []
    for imgfile in raw_imgfiles:
        img = fits.open(f"{raw_imgdir}/{imgfile}")[pos].data
        mins.append(np.nanmin(img))
        maxs.append(np.nanmax(img))
    full_min = np.nanmin(mins)
    full_max = np.nanmax(maxs)
    print(f'Min and max of all {len(raw_imgfiles)} OG files: {full_min, full_max}')

'''
Make normalized versions, save in raw dir
'''
norm_tag = 'full' if norm_type == 'full' else ''
if len([f for f in os.listdir(raw_imgdir) if f'{norm_tag}Norm' in f]) == 0:
    print(f'Saving full-series normalized versions of {len(raw_imgfiles)} OG files')
    count = 0
    for imgfile in raw_imgfiles:
        img = fits.open(f"{raw_imgdir}/{imgfile}")[pos].data
        header = fits.open(f"{raw_imgdir}/{imgfile}")[0].header
        if norm_type == 'individual':
            img_norm = (img - np.nanmin(img))/(np.nanmax(img) - np.nanmin(img))
        else:
            img_norm = (img - full_min)/(full_max - full_min)
        if np.max(img_norm) > 1:
            raise ValueError('Data has not been normalized')
        hdu = fits.PrimaryHDU(img_norm)
        hdu.writeto(f"{raw_imgdir}/{norm_tag}Norm_{imgfile}", overwrite=True)
        count += 1
    print(f'Saved {count} files (now {len([file for file in os.listdir(raw_imgdir) if file.startswith(f"{norm_tag}Norm")])} normed files)')

'''
Make subsections, analogously to for non-normed images, and place in subsection dir (e.g. UNetData_MURaM/fullNorm_images/)
'''
normsubsecdir = f'{UNetData_dir}/{norm_tag}norm_images/'  
print(f'Creating {normsubsecdir}')
if not os.path.exists(normsubsecdir): 
    os.mkdir(normsubsecdir) 
normfiles = [file for file in os.listdir(raw_imgdir) if file.startswith(f'{norm_tag}Norm')] # files = [fil for file in os.listdir(OG_dir) if file.startswith('Norm_VBI') and file.endswith('_4096') and not any(tag in file for tag in ['31_21', '33_13', '04_29', '17_20', '24_26'])]   
if len(os.listdir(normsubsecdir)) == 0:
    pad = int(np.shape(fits.open(f'{raw_imgdir}/{raw_imgfiles[0]}')[0].data)[0]/200)
    for normfile in normfiles:
        if 'MURaM' in UNetData_dir: tag = normfile[normfile.find('out')+4: normfile.find('.fits')]
        else: tag = normfile[normfile.find('VBI'):-5]
        print(f'')
        if os.path.exists(f'{normsubsecdir}/{tag}__{n**2-1}.npy'): # last .npy created from normfile
            print(f'{normsubsecdir}/{tag}__{n**2-1}.npy exists')
            print(f'\tNormed file {normfile} (tag {tag}) --> subsections already saved')
        else:
            print(f'\tNormed file {normfile} (tag {tag}) --> saving subsections')
            data = fits.open(f'{raw_imgdir}/{normfile}')[0].data 
            data = data[pad:-pad, pad:-pad] # Remove zero padding at edges of seg file
            N = np.min(np.shape(data)) 
            length = int(N/n) 
            num = 0
            for i in range(n):
                for j in range(n):
                    name = tag+'__'+str(num); num += 1
                    x1 = length * i
                    x2 = length * (i + 1)
                    y1 = length * j
                    y2 = length * (j + 1)
                    savedata = data[x1:x2, y1:y2]
                    np.save(normsubsecdir+name, savedata)
    normnpyfiles = [file for file in os.listdir(normsubsecdir) if '.npy' in file] # all files that aren't folders (train and test)
    print(f'Saved {len(normnpyfiles)} total subsection npy files, each with shape {np.shape(np.load(f"{normsubsecdir}/{normnpyfiles[0]}"))}')
    print(f'Make sure this matches the {len(os.listdir(f"{UNetData_dir}/images/train") + os.listdir(f"{UNetData_dir}/images/val"))} in old subsec dir') # no idea why there are a few more norm npy files for all_dkist... but shouldnt matter becasue just won't be moved to train and val folders right?
    
'''
Create non-TS UNetData subsec set
Move into train and val based on where non-normed counterpart is (note: some may not be moved if left some non-norm)
'''
oldsubsecdir = f'{UNetData_dir}/images' # for checking which folder it should go in
normnpyfiles = [file for file in os.listdir(normsubsecdir) if '.npy' in file] # all files that aren't folders (train and test)
if os.path.exists(f'{normsubsecdir}/train') == False: os.mkdir(f'{normsubsecdir}/train'); os.mkdir(f'{normsubsecdir}/val')
for normnpyfile in normnpyfiles:
    oldnpyfile = normnpyfile # names are the same (xxxxx__x.npy)
    if os.path.exists(f'{oldsubsecdir}/train/{oldnpyfile}'): 
        shutil.move(f'{normsubsecdir}/{normnpyfile}', f'{normsubsecdir}/train/{normnpyfile}')
    elif os.path.exists(f'{oldsubsecdir}/val/{oldnpyfile}'):
        shutil.move(f'{normsubsecdir}/{normnpyfile}', f'{normsubsecdir}/val/{normnpyfile}')
print(f'Moved {len(os.listdir(f"{normsubsecdir}/train") + os.listdir(f"{normsubsecdir}/val"))} files ({len(os.listdir(f"{normsubsecdir}/train"))} into train and {len(os.listdir(f"{normsubsecdir}/val"))} into val)')
print(f'Make sure this matches {len(os.listdir(f"{oldsubsecdir}/train") + os.listdir(f"{oldsubsecdir}/val"))} total non-normed npy files ({len(os.listdir(f"{oldsubsecdir}/train"))} in train and {len(os.listdir(f"{oldsubsecdir}/val"))} in val)')
fig, axs = plt.subplots(10,2, figsize=(7, 30))
for i in range(10):
    im = axs[i,0].imshow(np.load(f'{oldsubsecdir}/train/'+os.listdir(f'{oldsubsecdir}/train')[i])); plt.colorbar(im, ax=axs[i,0])
    im = axs[i,1].imshow(np.load(f'{normsubsecdir}/train/'+os.listdir(f'{normsubsecdir}/train')[i])); plt.colorbar(im, ax=axs[i,1])
    axs[i,0].xaxis.set_tick_params(labelbottom=False); axs[i,0].yaxis.set_tick_params(labelleft=False); axs[i,1].set_xticks([]); axs[i,1].set_yticks([])
plt.savefig('check_normdata')

'''
SEEMS LIKE PREVIOUS SETS WERE ACTUALLY NOT QUITE RIGHT (just like 20 instead of 21 imgs, so should be fine, but still best fix that before proceeeding)
Create TS UNetData subsec set
    - Make sure to keep correspondence with UNetData_MURaM_TSeries*/seg_images (tags must indicate the same image, and TTS must be the same)
'''

# # Make new dirctories for normed TSeries data 
# sizes = [21, 41] # 40 should have 41 imgs per set, e.g. 20 imgs (40 sec, for MURaM) on either side of target
# for size in sizes:
#     UNetTSData_dir = f"../../Data/UNetData_MURaM_TSeries{size-1}"  
#     normTSsubsecdir = f"{UNetTSData_dir}/{norm_tag}norm_images"; trnimgdir = f"{normsubsecdir}/train"; tstimgdir = f"{normsubsecdir}/val"; #segdir = f"{fromparent}/seg_images"; trnsegdir = f"{segdir}/train"; tstsegdir = f"{segdir}/val"
#     for dir in [normTSsubsecdir, trnimgdir, tstimgdir]:
#         if os.path.exists(dir)==False: 
#             print(f'Creating {dir}')
#             os.mkdir(dir)

#     # Temporarily copy all normed npy subsections into UNetData_*_TSeries*/{tag}norm_images/ [include parent dir in case some not in train or val] 
#     all_muram_norm_npy = [f for f in os.listdir(normsubsecdir) if '.npy' in f] #[f for f in os.listdir(f"{unetdata_dir}/{tag}norm_images") if '.npy' in f] + [f"../../Data/UNetData_MURaM/{tag}norm_images/train/{f}" for f in os.listdir(f"../../Data/UNetData_MURaM/{tag}norm_images/train")] + [f"../../Data/UNetData_MURaM/{tag}norm_images/val/{f}" for f in os.listdir(f"../../Data/UNetData_MURaM/{tag}norm_images/val")]
#     for file in all_muram_norm_npy:
#         shutil.copy(f'{normsubsecdir}/{file}', f"{normTSsubsecdir}/{file}") #     if os.path.exists(f"{parent}/norm_images/{file}") == False:

#     # Split imgs into and train and test based on seg location, keeping sections together as cubes
#     segTSsubsecdir = f'../../Data/UNetData_MURaM_TSeries{size-1}/seg_images' # for checking which train, test folder
#     normsubsecfiles = [file for file in os.listdir(normTSsubsecdir) if '.npy' in file]; #segfiles = [file for file in os.listdir(newparent) if '.npy' in file and ('SEG' in file)==True]
#     shape = np.load(f"../../Data/UNetData_MURaM/images/train/000130__3.npy").shape # example image 
#     for region in range(n**2): # for each spatial region
#         files_region = np.sort([file for file in normsubsecfiles if f"__{region}" in file]); # segfiles_region = [file for file in segfiles if f"__{region}" in file]
#         print(f'files_region {files_region}')
#         start_step = 0
#         while start_step <= 1801-size: # for each set of size imgs (there are 1801 total)
#             #seg = np.load(f"{newparent}/{segfiles_region[start_step + int((size-1)/2)]}") # seg for the target image
#             files_set = files_region[start_step:start_step+size]
#             print(f'files_set {files_set}')
#             imgset = np.zeros((size, shape[0], shape[1]))
#             for i in range(size-1):
#                 print(f'   loading {normsubsecdir}/{files_set[i]}')
#                 imgset[i,:,:] = np.load(f"{normsubsecdir}/{files_set[i]}") # BUT HOW DID I KNOW THIS WOULD BE IN ORDER WITHOUT SORTING IVE NOW APPLIED ABOVE TO IMGFILES_REGION??
#             imgsetname = f"{files_set[0][0:-7]}to{files_set[-1][0:-7]}_{region}"
#             segname = f"SEG_{imgsetname}.npy"
#             print(f'imgsetname {imgsetname}, segname {segname}')
#             if segname in os.listdir(f"{segTSsubsecdir}/train/"): 
#                 folder = 'train'
#             elif segname in os.listdir(f"{segTSsubsecdir}/val/"):
#                 folder = 'val'
#             else:
#                 raise ValueError(f'Corresponding segmented image {segname} not found')
#             np.save(f"{normTSsubsecdir}/{folder}/{imgsetname}", imgset); # np.save(f"{newparent}/SEG_images/{folder}/{segname}", seg)
#             start_step += size   

#     # Checks
#     if len(os.listdir(f'{normTSsubsecdir}/train')) == len(os.listdir(f'{UNetTSData_dir}/images/train')) == len(os.listdir(f'{segTSsubsecdir}/train')):
#         print('Good! Train directories that should correpond match in length')
#     else:
#         raise ValueError('Train directories that should correpond DO NOT match in length')
#     if len(os.listdir(f'{normTSsubsecdir}/val')) == len(os.listdir(f'{UNetTSData_dir}/images/val')) == len(os.listdir(f'{segTSsubsecdir}/val')):
#         print('Good! Test directories that should correpond match in length')
#     else:
#         raise ValueError('Test directories that should correpond DO NOT match in length')
#     fig, axs = plt.subplots(3,2, figsize=(7, 10))
#     normingsets = os.listdir(f"{normTSsubsecdir}/train")
#     for i in range(3):
#         idx = int(np.random.choice(np.linspace(0,len(normingsets),len(normingsets))))
#         normimgset = np.load(f"{normTSsubsecdir}/train/{normingsets[idx]}")
#         imgset = np.load(f"{UNetTSDatadir}/images/train/{normingsets[idx]}") 
#         im0 = axs[i,0].imshow(imgset[2]); plt.colorbar(im0, ax=axs[i,0])
#         im1 = axs[i,1].imshow(normimgset[2]); plt.colorbar(im1,  ax=axs[i,1])
#     axs[0,0].set_title('non-normed')
#     axs[0,1].set_title('normed')
#     plt.savefig('check_normTSdata')
                  
#     # Remove temporarily copy all normed npy subsections
#     for file in [file for file in os.listdir(normTSsubsecdir) if '.npy' in file]:
#         os.remove(f"{parent}/{norm_tag}norm_images/{file}")
   
'''
Redo med8 images with normed data (and just delete med30)
Only saving for train set - just calcuate on-the-fly for test
'''

os.mkdir(f'{UNetData_dir}/med8_images')
os.mkdir(f'{UNetData_dir}/med8_images/train')
for img in os.listdir(f'{UNetData_dir}/images/train/'):
    data = np.load(f'{UNetData_dir}/{norm_tag}norm_images/train/{img}') # why was I using non-norm and then normalizing individual image? isnt that what I was tyring to aviod???
    med = sndi.median_filter(data, size=8)
    name = f'med8_{img}'
    np.save(f'{UNetData_dir}/med8_images/train/{name}', med)