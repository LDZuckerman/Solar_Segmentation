
import os, shutil
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sndi

'''
Params
'''
types = ['DVmag'] #'med8', mag', 'vx', 'vy', 'vz'
UNetData_dir = '../../Data/UNetData_MURaM'
norm_type = 'full' #'individual'
n = 3 # sqrt of num pieces to break each image into
pos = 0

for typ in types:
    
    print(f'Making {typ} set with {norm_type} normalization', flush=True)
    norm_tag = 'full' if norm_type == 'full' else ''    

    if typ != 'med8s':
        
        raw_newdir = f'../../Data/MURaM_{typ}' # The directory in which the raw files of the new type are already stored
        files = os.listdir(raw_newdir) #[filename for filename in os.listdir(raw_newdir) if filename.startswith('tau')]
        startswith = 'tau' if typ in ['mag', 'vx', 'vy','vz'] else ''
        raw_files = [f for f in files if 'orm' not in f]
        
        # Make normalized versions of all the typ imgs in OG set
        if len([f for f in os.listdir(raw_newdir) if f.startswith(f'{norm_tag}Norm')]) != len([f for f in os.listdir(raw_newdir) if f.startswith(startswith)]) and len([f for f in os.listdir(raw_newdir) if f.startswith(startswith)])!=0 :
            
            # Find full series min and max
            if norm_type == 'full':
                mins = []
                maxs = []
                for file in raw_files:
                    try:
                        im = fits.open(f"{raw_newdir}/{file}")[pos].data
                    except OSError as E:
                        print(f'Error {E} for file {file}\nAll files: {files}')
                    mins.append(np.nanmin(im))
                    maxs.append(np.nanmax(im))
                full_min = np.nanmin(mins)
                full_max = np.nanmax(maxs)
                print(f'Min and max of all {len(files)} OG {typ} files: {full_min, full_max}', flush=True)
            
            # Save normalized versions
            norm_tag = 'full' if norm_type == 'full' else ''
            print(f'Saving {norm_type} normalized versions of {len(files)} OG files', flush=True)
            count = 0
            for file in raw_files:
                im = fits.open(f"{raw_newdir}/{file}")[pos].data
                header = fits.open(f"{raw_newdir}/{file}")[0].header
                if norm_type == 'individual':
                    im_norm = (im - np.nanmin(im))/(np.nanmax(im) - np.nanmin(im))
                elif norm_type == 'full':
                    im_norm = ((im - full_min)/(full_max - full_min))
                else:
                    raise ValueError(f'norm_type must be either "full" or "individual" not {norm_type}')
                hdu = fits.PrimaryHDU(im_norm)
                hdu.writeto(f"{raw_newdir}/{norm_tag}Norm_{file}", overwrite=True)
                count += 1
            print(f'Saved {count} files (now {len([file for file in os.listdir(raw_newdir) if file.startswith(f"{norm_tag}Norm")])} normed files)', flush=True)    
        else:
            print(f'Already saved normed versions of raw {typ} imgs into raw {typ} image dir {raw_newdir}', flush=True)

        # Create UNetData subsec set
        files = [filename for filename in os.listdir(raw_newdir) if filename.startswith(f'{norm_tag}Norm')]
        newsubsecdir = f'../../Data/UNetData_MURaM/{norm_tag}norm_{typ}_images'
        print(f'Creating {newsubsecdir}', flush=True)
        if not os.path.exists(newsubsecdir): 
            os.mkdir(newsubsecdir)  
        for file in files:
            end_token = 'Bz' if typ == 'mag' else typ if typ in ['vx','vy','vz'] else 'fits' if typ == 'DVmag' else ''
            strt_token = 'Norm' if typ == 'DVmag' else 'slice'
            tag = file[file.find(strt_token)+12: file.find(end_token)-1] # e.g. "016690" from "tau_slice_1.000.016690_Bz.fits.gz" or from "fullNorm_predBz_016690.fits"
            if os.path.exists(f'{newsubsecdir}/{tag}__{n**2-1}.npy'): # last .npy created from file
                #print(f'{newsubsecdir}/{tag}__{n**2-1}.npy exists')
                print(f'\tNormed {typ} file {file} (tag {tag}) -> Subsections already saved', flush=True)
            elif os.path.exists(f'{newsubsecdir}/train/{tag}__{n**2-1}.npy') or os.path.exists(f'{newsubsecdir}/val/{tag}__{n**2-1}.npy'): # last .npy created from normfile
                print(f'\tNormed {typ} file {file} (tag {tag}) --> Subsections already saved and split into train/test', flush=True)
            else:
                print(f'\tNormed {typ} file {file} (tag {tag}) -> Saving subsections', flush=True)
                data = fits.open(f'{raw_newdir}/{file}')[0].data 
                # Remove zero padding at edges of seg file
                pad = int(np.shape(data)[0]/200)
                data = data[pad:-pad, pad:-pad]
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
                        np.save(f'{newsubsecdir}/{name}', savedata)

        # Seperate into train and test folders 
        imgsubsecdir = f"{UNetData_dir}/{norm_tag}norm_images" # for checking which folder it should go in, could also use seg or norm_images, or anything
        newfiles = [file for file in os.listdir(newsubsecdir) if '.npy' in file]
        for d in [f'{newsubsecdir}/train', f'{newsubsecdir}/val']:
            if not os.path.exists(d):
                os.mkdir(d)
        for newfile in newfiles:
            imgfile = newfile # names are the same (xxxxx__x.npy)
            if os.path.exists(f'{imgsubsecdir}/train/{imgfile}'): 
                shutil.move(newsubsecdir+'/'+newfile, newsubsecdir+'/train/'+newfile)
            elif os.path.exists(f'{imgsubsecdir}/val/{imgfile}'):
                shutil.move(newsubsecdir+'/'+newfile, newsubsecdir+'/val/'+newfile)
            else: 
                #raise ValueError(f'Subsection {imgfile} not in train or test set')
                print(f'WARNING: subsection {imgfile} not in train or test set ')
        print(f'Moved {len(os.listdir(f"{newsubsecdir}/train") + os.listdir(f"{newsubsecdir}/val"))} files ({len(os.listdir(f"{newsubsecdir}/train"))} into train and {len(os.listdir(f"{newsubsecdir}/val"))} into val)', flush=True)
        print(f'Make sure this matches {len(os.listdir(f"{imgsubsecdir}/train"))} and {len(os.listdir(f"{imgsubsecdir}/val"))} in corresponding image set train and val)', flush=True)
        fig, axs = plt.subplots(10,2, figsize=(7, 30))
        for i in range(10):
            im = axs[i,0].imshow(np.load(f'{imgsubsecdir}/train/'+os.listdir(f'{imgsubsecdir}/train')[i])); plt.colorbar(im, ax=axs[i,0])
            im = axs[i,1].imshow(np.load(f'{newsubsecdir}/train/'+os.listdir(f'{newsubsecdir}/train')[i])); plt.colorbar(im, ax=axs[i,1])
            axs[i,0].xaxis.set_tick_params(labelbottom=False); axs[i,0].yaxis.set_tick_params(labelleft=False); axs[i,1].set_xticks([]); axs[i,1].set_yticks([])
        plt.savefig(f'check_norm{typ}data')
        # NOTE - after this runs, should definitely check that nothing has gone wrong and loaded images in new and img set do correspond
    
    elif typ == 'med8': 
        # NOTE: I am saving median filters of the (full) normed images but NOT (full) normed median filtered images!
        #       In NNs, want to use residual (img - med), so before doing that need to compute min and max of full sequence residuals!
        os.mkdir(f'{UNetData_dir}/{norm_tag}norm_med8_images')
        os.mkdir(f'{UNetData_dir}/{norm_tag}norm_med8_images/train')
        for img in os.listdir(f'{UNetData_dir}/{norm_tag}norm_images/train/'):
            data = np.load(f'{UNetData_dir}/{norm_tag}norm_images/train/{img}') 
            med = sndi.median_filter(data, size=8)
            name = f'med8_{img}'
            np.save(f'{UNetData_dir}/{norm_tag}norm_med8_images/train/{name}', med)

        