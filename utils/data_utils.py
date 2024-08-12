import numpy as np
import cv2
import sunpy
import scipy.ndimage as sndi
import pandas as pd
from sklearn import preprocessing
import astropy.io.fits as fits 
import os
import matplotlib.pyplot as plt
import skimage as sk
import scipy.stats as stats
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


###############
# Dataset class
###############

class MyDataset(Dataset):
    '''
    Dataset class for loading images and labels from a directory.
    '''
    def __init__(self, image_dir, mask_dir, set, norm=False, channels=['X'], n_classes=2, randomSharp=False, im_size=None):
        self.image_dir = f'{image_dir}{set}'
        self.mask_dir = f'{mask_dir}{set}'
        self.set = set
        self.images = os.listdir(f'{image_dir}{set}')
        self.norm = norm
        self.channels = channels
        self.n_classes = n_classes
        self.randomSharp = eval(str(randomSharp))
        self.im_size = im_size
        self.resize = transforms.Resize(im_size, antialias=None)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Get image
        img_path = os.path.join(self.image_dir, self.images[index]) # path to one data image (or SET of [20, npix, npix] if timeseries)
        img = np.load(img_path)
        if np.max(img) > 1:
            raise ValueError('This image does not appear to come from a pre-normalized set')
        if img.dtype.byteorder == '>':
            img = img.newbyteorder().byteswap() 
        if self.randomSharp: # add 50% chance of image being blurred/sharpened by a factor pulled from a skewed guassian (equal chance of 1/4 and 4)
            img =((img - np.nanmin(img))/(np.nanmax(img) - np.nanmin(img))) # first must [0, 1] normalize
            img = torch.from_numpy(np.expand_dims(img, axis=0)) # transforms expect a batch dimension
            n = stats.halfnorm.rvs(loc=1, scale=1, size=1)[0]
            s = n if np.random.rand(1)[0] < 0.5 else 1/n
            try: 
                transf = transforms.RandomAdjustSharpness(sharpness_factor=s, p=0.5, antialias=None)
            except Exception: # If running with older torch version
                transf = transforms.RandomAdjustSharpness(sharpness_factor=s, p=0.5)
            img = transf(img)[0] # remove batch dimension for now
        if self.im_size != None: # cut to desired size, e.g. to make divisible by 2 5 times, for WNet
            img = np.array(self.resize(torch.from_numpy(np.expand_dims(img, axis=0)))).squeeze()
        if self.channels != ['X']: # Add feature layers
            if self.channels[0].startswith('timeseries'):
                tag = self.channels[0][self.channels[0].find('ies')+3:]
                image = fill_timeseries(img, tag)
            else:
                image = np.zeros((len(self.channels), img.shape[0], img.shape[1]), dtype=np.float32) # needs to be float32 not float64
                image[0, :, :] = img
                for i in range(1, len(self.channels)):
                    image[i, :, :] = get_feature(img, self.channels[i], index, self.images[index], img_path, self.resize, self.set)
        else: # Add dummy axis
            image = np.zeros((1, img.shape[0], img.shape[1]), dtype=np.float32) # needs to be float32 not float64
            image[0, :, :] = img
        # Get labels
        mask_path = os.path.join(self.mask_dir, 'SEG_'+self.images[index]) # path to one labels image (use name, not idx, for safety)
        labels = np.load(mask_path).newbyteorder().byteswap() 
        if self.n_classes==4: # One-hot encode targets so they are the correct size
            mask = np.zeros((4, labels.shape[0], labels.shape[1]), dtype=np.float32) # needs to be float32 not float64
            mask_gr, mask_ig, mask_bp, mask_dm = np.zeros_like(labels), np.zeros_like(labels), np.zeros_like(labels), np.zeros_like(labels)
            mask_ig[labels == 0] = 1 # 1 where intergranule, 0 elsewhere
            mask_dm[labels == 0.5] = 1 # 1 where dim middle, 0 elsewhere
            mask_gr[labels == 1] = 1 # 1 where granule, 0 elsewhere
            mask_bp[labels == 1.5] = 1 # 1 where bright point, 0 elsewhere
            mask[0, :, :] = mask_ig
            mask[1, :, :] = mask_dm
            mask[2, :, :] = mask_gr
            mask[3, :, :] = mask_bp
        if self.n_classes==3: # One-hot encode targets so they are the correct size
            mask = np.zeros((3, labels.shape[0], labels.shape[1]), dtype=np.float32) # needs to be float32 not float64
            mask_gr, mask_ig, mask_bp = np.zeros_like(labels), np.zeros_like(labels), np.zeros_like(labels)
            mask_ig[labels == 0] = 1 # 1 where intergranule, 0 elsewhere
            mask_gr[labels == 1] = 1 # 1 where granule, 0 elsewhere
            mask_bp[labels == 1.5] = 1 # 1 where bright point, 0 elsewhere
            mask[0, :, :] = mask_ig
            mask[1, :, :] = mask_gr
            mask[2, :, :] = mask_bp
        elif self.n_classes==2: # One-hot encode targets so they are the correct size
            mask = np.zeros((2, labels.shape[0], labels.shape[1]), dtype=np.float32) # needs to be float32 not float64
            mask_gr, mask_ig = np.zeros_like(labels), np.zeros_like(labels)
            mask_ig[labels == 0] = 1 # 1 where intergranule, 0 elsewhere
            mask_gr[labels == 1] = 1 # 1 where granule, 0 elsewhere
            mask[0, :, :] = mask_ig
            mask[1, :, :] = mask_gr
        if self.im_size != None: # cut to desired size, e.g. to make divisible by 2 4 times, for WNet
            mask = np.array(self.resize(torch.from_numpy(mask)))

        return image, mask


#######################################
# Helper functions for creating dataset
#######################################

def fill_timeseries(img, tag):
    '''
    Create a timeseries of images around the target image.
    '''

    if tag == '20_5':
        image = np.zeros((5, img.shape[1], img.shape[2]), dtype=np.float32) # needs to be float32 not float64
        image[0, :, :] = img[0, :, :]
        image[1, :, :] = img[5, :, :]
        image[2, :, :] = img[10, :, :] # target image (is it important to put this here?)
        image[3, :, :] = img[15, :, :]
        image[4, :, :] = img[19, :, :] # should've probabaly done sets of 21.. oh well
    elif tag == '40_5':
        image = np.zeros((5, img.shape[1], img.shape[2]), dtype=np.float32) # needs to be float32 not float64
        image[0, :, :] = img[0, :, :]
        image[1, :, :] = img[10, :, :]
        image[2, :, :] = img[20, :, :] # target image (is it important to put this here?)
        image[3, :, :] = img[30, :, :]
        image[4, :, :] = img[40, :, :] # these sets are 41
    elif tag == '80_5':
        image = np.zeros((5, img.shape[1], img.shape[2]), dtype=np.float32) # needs to be float32 not float64
        image[0, :, :] = img[0, :, :]
        image[1, :, :] = img[20, :, :]
        image[2, :, :] = img[40, :, :] # target image (is it important to put this here?)
        image[3, :, :] = img[60, :, :]
        image[4, :, :] = img[80, :, :] # these sets are 81
    elif tag == '40_9':
        image = np.zeros((9, img.shape[1], img.shape[2]), dtype=np.float32) # needs to be float32 not float64
        image[0, :, :] = img[0, :, :]
        image[1, :, :] = img[5, :, :]
        image[2, :, :] = img[10, :, :] 
        image[3, :, :] = img[15, :, :]
        image[4, :, :] = img[20, :, :] # target image (is it important to put this here?)
        image[5, :, :] = img[25, :, :]
        image[6, :, :] = img[30, :, :]
        image[7, :, :] = img[35, :, :] 
        image[8, :, :] = img[40, :, :] # these sets are 41
    else: 
        raise ValueError(f'Timeseries tag {tag} not recognized')
    
    return image

def get_feature(img, name, index, image_name, imgpath, resize, set=None):
    '''
    Add desired feature layers to the image.
    '''
    if name == 'gradx': a = np.gradient(img)[0]
    elif name == 'grady': a = np.gradient(img)[1]
    elif name == 'smoothed': a = scipy.ndimage.gaussian_filter(img, sigma=3)
    elif '**' in name:
        n = int(name[-1])
        a = img**n
    elif name == 'Bz':
        mag_path = f'../Data/UNetData_MURaM/mag_images/{set}/{image_name}' # path to one mag image 
        mag = np.load(mag_path).newbyteorder().byteswap()
        if mag.shape != img.shape: # if img_size != None above, so cut img to desired size, need to do that for mag
            mag = np.array(resize(torch.from_numpy(np.expand_dims(mag, axis=0)))).squeeze()
        mag = mag**2
        a = (mag - np.mean(mag))/np.std(mag) # normalize to std normal 
    # elif name == 'binary_residual':
    #     im_scaled = (img-np.min(img))/(np.max(img)-np.min(img))
    #     if set == 'train':
    #         if os.path.exists(f'../NN_outputs/WNet8m_outputs/predict_on_train/wnet8seg_{index}'): # if already save
    #             wnet8_preds = np.squeeze(np.load(f'../NN_outputs/WNet8m_outputs/predict_on_train/wnet8seg_{index}'))
    #         else: 
    #             model = MyWNet(squeeze=2, ch_mul=64, in_chans=2, out_chans=2)
    #             model.load_state_dict(torch.load(f'../NN_storage/WNet8m.pth'))
    #             x = np.zeros((1, 2, img.shape[0], img.shape[1]), dtype=np.float32) # needs to be float32 not float64
    #             x[0, 0, :, :] = img
    #             x[0, 1, :, :] = img**2
    #             X = torch.from_numpy(x) # transforms.Resize(128)(torch.from_numpy(x))
    #             probs = model(X, returns='enc') # defualt is to return dec, but we want seg
    #             wnet8_preds = np.argmax(probs.detach().numpy(), axis=1).astype(float).squeeze() # turn probabilities [n_obs, n_class, n_pix, n_pix] into predicted class labels [n_obs, n_pix, n_pix]
    #             np.save(f'../NN_outputs/WNet8m_outputs/predict_on_train/wnet8seg_train/wnet8seg_{index}', wnet8_preds)
    #     if set == 'test':
    #         wnet8_preds = np.squeeze(np.load(f'../NN_outputs/WNet8m_outputs/pred_{index}.npy'))
    #     kernel = np.ones((30,30))/900
    #     wnet8_preds_smooth = cv2.filter2D(wnet8_preds, -1, kernel)
    #     a = (wnet8_preds_smooth - im_scaled)**2 
    elif name == 'median_residual':
        meddir = f'{imgpath[0:imgpath.find("norm_images/")]}med8_images/{set}/'
        if os.path.exists(meddir):
            med_path = f'{meddir}med8_{image_name}'
            med = np.load(med_path)
            if np.shape(med) != np.shape(img): # already cut down image
                med = np.array(transform(torch.from_numpy(np.expand_dims(med, axis=0)))).squeeze()
            a = img - med
        else: 
            if set == 'val': # didint save for test set - instead just compute here
                s = 8
                a = img - sndi.median_filter(img, size=s)
            else:
                raise FileNotFoundError(f'Median filtered images not saved for this set ({meddir} does not exist).')
        if 'MURAM' in imgpath: 
            mean = 0.000368; sd = 0.019967
            a = (a - mean)/sd
    else: raise ValueError(f'Channel name {name} not recognized')

    return a

def check_inputs(train_ds, train_loader, savefig=False, name=None):
    '''
    Check data is loaded correctly
    '''
    print('Train data:')
    print(f'     {len(train_ds)} obs, broken into {len(train_loader)} batches')
    train_features, train_labels = next(iter(train_loader))
    shape = train_features.size()
    print(f'     Each batch has data of shape {train_features.size()}, e.g. {shape[0]} images, {[shape[2], shape[3]]} pixels each, {shape[1]} layers (features)')
    shape = train_labels.size()
    print(f'     Each batch has labels of shape {train_labels.size()}, e.g. {shape[0]} images, {[shape[2], shape[3]]} pixels each, {shape[1]} layers (classes)')
    if savefig:
        fig, axs = plt.subplots(7, 3, figsize=(3*4, 7*4))
        axs[0,0].set_title('image[0]')
        #ax2.set_title('image[1]')
        axs[0,2].set_title('labels')
        for i in range(7):
            X, y = next(iter(train_loader))
            y = onehot_to_map(y)
            im1 = axs[i,0].imshow(X[0,0,:,:], vmin=0, vmax=1); plt.colorbar(im1, ax=axs[i,0]) # first img in batch, first channel
            if X.shape[1] > 1: im2 = axs[i,1].imshow(X[0,1,:,:], vmin=0, vmax=1); plt.colorbar(im2, ax=axs[i,1]) # first img in batch, first channel
            im3 = axs[i,2].imshow(y[0,:,:]); plt.colorbar(im3, ax=axs[i,2]) # first y in batch, already class-collapsed
        plt.savefig(f'traindata_{name}'); a=b


###############
# Preprocessing 
###############

def histogram_equalization(data, n_bins=256):
   '''
   Perform histogram equalization on the image data.
   '''
   #get image histogram
   hist, bins = np.histogram(data.flatten(), n_bins, normed=True)
   cdf = hist.cumsum()
   cdf = 255 * cdf / cdf[-1] #normalize
   # use linear interpolation of cdf to find new pixel values
   data_out = np.interp(data.flatten(), bins[:-1], cdf)
   data_out = data_out.reshape(data.shape)

   return data_out

def match_to_firstlight(obs_data, n_bins=2000):
    '''
    Match the histogram of true observed data to the first light data.
    '''
    # Read in "good" image and normalize
    synth_data = fits.open('../Data/DKIST_example.fits')[0].data
    synth_data = (synth_data-np.nanmean(synth_data))/np.nanstd(synth_data)
    # Compute good image histogram and cdf
    synth_bins = np.linspace(np.nanmin(synth_data), np.nanmax(synth_data), n_bins)
    width_synth_bins = synth_bins[1]-synth_bins[0]
    synth_hist, _ = np.histogram(synth_data[np.isfinite(synth_data)].flatten(), bins=synth_bins, density=True)
    synth_hist = np.expand_dims(synth_hist, axis=0) # since not loading data as stacked array
    synth_cdf = np.cumsum(np.nanmean(synth_hist*width_synth_bins, axis=0))
    # Normalize "bad" image
    obs_data = (obs_data-np.nanmean(obs_data))/np.nanstd(obs_data)
    # Compute good image histogram and cdf
    obs_bins = np.linspace(np.nanmin(obs_data), np.nanmax(obs_data), n_bins)
    width_obs_bins = obs_bins[1]-obs_bins[0]
    obs_hist, _ = np.histogram(obs_data[np.isfinite(obs_data)].flatten(), bins=obs_bins, density=True)
    obs_hist = np.expand_dims(obs_hist, axis=0) # since not loading data as stacked array
    obs_cdf = np.cumsum(np.nanmean(obs_hist*width_obs_bins, axis=0))
    # Perform matching
    obs_data_matched = hist_matching(obs_data, obs_cdf, obs_bins, synth_cdf, synth_bins)
    obs_data_matched = obs_data_matched.reshape(np.shape(obs_data))

    return obs_data_matched
    
def hist_matching(data_in, cdf_in, bins_in, cdf_out, bins_out):

    bins_in = 0.5*(bins_in[:-1] + bins_in[1:]) # Points for interpolation (input bins contain the edges)
    bins_out = 0.5*(bins_out[:-1] + bins_out[1:])
    cdf_tmp = np.interp(data_in.flatten(), bins_in.flatten(), cdf_in.flatten())  # Interpolation
    data_out = np.interp(cdf_tmp, cdf_out.flatten(), bins_out.flatten())
    return data_out

def pre_proccess(data, labels, gradientFeats=True, kernalFeat=True):

    # Flatten features and labels
    dataflat = data.reshape(-1)
    labelsflat = labels.reshape(-1)
    # Put features and labels into df
    df = pd.DataFrame()
    df['OG_value'] = dataflat
    df = add_kernel_feats(df, dataflat) # Add values of different filters as features
    df = add_gradient_feats(df, data) # Add value of gradient as feature
    df['labels'] =  labelsflat
    # Make X and Y
    X =  df.drop(labels =["labels"], axis=1)
    Y = df['labels']
    Y = preprocessing.LabelEncoder().fit_transform(Y) # turn floats 0, 1, to categorical 0, 1
    return X, Y

def post_process(preds, data=None):

    preds = np.copy(preds).astype(float)  # Float conversion for correct region numbering.
    preds2 = np.ones_like(preds)*20 # Just to aviod issues later on 
    # If its a 2-value seg
    if len(np.unique(preds)) == 2:
        # Assign a number to each predicted region
        labeled_preds = skimage.measure.label(preds + 1, connectivity=2)
        values = np.unique(labeled_preds)
        # Find numbering of the largest region (IG region)
        size = 0
        for value in values:
            if len((labeled_preds[labeled_preds == value])) > size:
                IG_value = value
                size = len(labeled_preds[labeled_preds == value])
        # Where labeled_preds=IG_value set preds2 to zero, otherwise 1
        preds2[labeled_preds == IG_value] = 0
        preds2[labeled_preds != IG_value] = 1 
    # If its a 3-value seg
    elif len(np.unique(preds)) == 3:
        # WAIT BUT THIS WONT HELP ANYTHING CUASE NONE OF THE 3-VALUE ONES ID BPSs
        # NEED AN ALGORITHM TO MERGE N CLUSTERS INTO 3
        # 
        # Find the seg value of the region corresponding to the lowest and highest avg pix value in the og data 
        highest_mean = 0
        lowest_mean = sum(data) # will never be higher than this
        for seg_val in np.unique(preds):
            if np.mean(data[preds == seg_val]) < lowest_mean:
                IG_value = seg_value
            if np.mean(data[preds == seg_val]) > highest_mean:
                BP_value = seg_value
        # Where labeled_preds=IG_value set preds2 to zero, where BP_value, 0.5, and else (granule), 1
        preds2[labeled_preds == IG_value] = 0
        preds2[labeled_preds == BP_value] = 0.5
        preds2[labeled_preds != IG_value and labeled_preds != BP_value] 
    else:
        print('NOT YET IMPLEMENTED: NEED TO FIND A WAY TO GET >3-VALUE SEG INTO FORM COMPARABLE TO LABELS'); a=b

    return preds2


#############################################
# Add simple features for traditional methods
#############################################

def add_kernel_feats(df, dataflat):

    df_new = df.copy()
    k1 = np.array([[1, 1, 1],  # blur (maybe useful?)
                [1, 1, 1],
                [1, 1, 1]])/9
    k2 = np.array([[0, -1, 0],  # sharpening (probably not useful?)
                [-1, 5, -1],
                [0, -1, 0]])
    k3 = np.array([[-1, -1, -1],  # edge detection (probably not useful?)
                [-1, 8, -1],
                [-1, -1, -1]])
    kernels = [k1, k2, k3]
    for i in range(len(kernels)):
        kernel = kernels[i]
        filtered_img = cv2.filter2D(dataflat, -1, kernel).reshape(-1) # filtered_img = cv2.GaussianBlur(data, (5,5), 0)#.reshape(-1)
        df_new['kernel'+str(i)] = filtered_img

    return df_new

def add_gradient_feats(df, data):

    df_new = df.copy()
    df_new['gradienty'] = np.gradient(data)[0].reshape(-1) # Attempted to use cv2.Laplacian, but require dtype uint8, and converting caused issues with the normalization #cv2.Laplacian(data,cv2.CV_64F).reshape(-1)
    df_new['gradientx'] = np.gradient(data)[1].reshape(-1)

    return df_new

def add_sharpening_feats(df, dataflat):

    df_new = df.copy()
    df_new['value2'] = dataflat**2
    
    return df_new

######################################################### 
# Algorithmic segmentation from DKISTSegmentation project 
# Some modifcations from SUNKIT-IMAGE version
#   - final class number is 0 (IG), 0.5 (DM), 1 (GR), 1.5 (BP)
#   - operate on np arrays not sunpy maps
#   - add mark_BP flag to mark bright points
#   - add pad argument to adjust pad
######################################################### 

def segment_array_v2(map, resolution, *, skimage_method="li", mark_dim_centers=False, mark_BP=True, bp_min_flux=None, bp_max_size=0.15, footprint=250, pad=None): 
    """
    Segment an optical image of the solar photosphere into four-value maps with:

     * 0 as intergranule
     * 0.5 as "dim-middle" (optional)
     * 1 as granule
     * 1.5 "brightpoint" (optional)

    Parameters
    ----------
    smap : `numpy.ndarray`
        NumPy array containing data to segment.
    resolution : `float`
        Spatial resolution (arcsec/pixel) of the data.
    skimage_method : {"li", "otsu", "isodata", "mean", "minimum", "yen", "triangle"}
        scikit-image thresholding method, defaults to "li".
        Depending on input data, one or more of these methods may be
        significantly better or worse than the others. Typically, 'li', 'otsu',
        'mean', and 'isodata' are good choices, 'yen' and 'triangle' over-
        identify intergranule material, and 'minimum' over identifies granules.
    mark_dim_centers : `bool`
        Whether to mark dim granule centers as a separate category for future exploration.
    mark_bright_points : `bool`
        Whether to mark bright points as a seperate catagory for future exploration
    bp_min_flux : `float`
        Minimum flux level per pixel for a region to be considered a Bright Point.
        Defaalt is one standard deviation above the mean flux.
    bp_max_size: `float`
        Maximum diameter (arcsec) to consider a region a Bright Point.
        Defualt of 0.15. 
    pad: `int`
        Number of pixels to remove from each edge of the image.
        Default is image length / 200

    Returns
    -------
    segmented_map : `numpy.ndarray`
        NumPy array containing a segmented image (with the original header).
    """

    # Obtain local histogram equalization of map.
    map_norm = ((map - np.nanmin(map))/(np.nanmax(map) - np.nanmin(map))) * 225 # min-max normalization to [0, 225] 
    map_HE = sk.filters.rank.equalize(map_norm.astype(int), footprint=sk.morphology.disk(footprint)) # MAKE FOOTPRINT SIZE DEPEND ON RESOLUTION!!!
    # Apply initial skimage threshold for initial rough segmentation into granules and intergranules.
    median_filtered = sndi.median_filter(map_HE, size=3)
    threshold = get_threshold_v2(median_filtered, skimage_method)
    segmented_image = np.uint8(median_filtered > threshold)
    # Fix the extra intergranule material bits in the middle of granules.
    seg_im_fixed = trim_intergranules_v2(segmented_image, mark=mark_dim_centers, pad=pad)
    # Mark faculae and get final granule and facule count.
    if mark_BP: seg_im_markfac, faculae_count, granule_count = mark_faculae_v2b(seg_im_fixed, map, map_HE, resolution, bp_min_flux, bp_max_size)
    else: seg_im_markfac = seg_im_fixed
    # logging.info(f"Segmentation has identified {granule_count} granules and {faculae_count} faculae")
    segmented_map = seg_im_markfac
    return segmented_map

def get_threshold_v2(data, method):
    """
    Get the threshold value using given skimage segmentation type.

    Parameters
    ----------
    data : `numpy.ndarray`
        Data to threshold.
    method : {"li", "otsu", "isodata", "mean", "minimum", "yen", "triangle"}
        scikit-image thresholding method.

    Returns
    -------
    threshold : `float`
        Threshold value.
    """
    
    if len(data.flatten()) > 2000**2:
        data = np.random.choice(data.flatten(), (500, 500))
        print(f'\tWARNING: data too big so computing threshold based on random samples reshaped to 500x500 image')

    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be an instance of a np.ndarray")
    elif method == "li":
        threshold = skimage.filters.threshold_li(data)
    if method == "otsu":
        threshold = skimage.filters.threshold_otsu(data)
    elif method == "yen":
        threshold = skimage.filters.threshold_yen(data)
    elif method == "mean":
        threshold = skimage.filters.threshold_mean(data)
    elif method == "minimum":
        threshold = skimage.filters.threshold_minimum(data)
    elif method == "triangle":
        threshold = skimage.filters.threshold_triangle(data)
    elif method == "isodata":
        threshold = skimage.filters.threshold_isodata(data)

    return threshold

def trim_intergranules_v2(segmented_image, mark=False, pad=None):
    """
    Remove the erroneous identification of intergranule material in the middle
    of granules that the pure threshold segmentation produces.

    Parameters
    ----------
    segmented_image : `numpy.ndarray`
        The segmented image containing incorrect extra intergranules.
        Will have small padding of zeros around edges.
    mark : `bool`
        If `False` (the default), remove erroneous intergranules.
        If `True`, mark them as 0.5 instead (for later examination).
    pad : `int`
        Number of pixels to remove from each edge of the image.
        Default is image length / 200

    Returns
    -------
    segmented_image_fixed : `numpy.ndarray`
        The segmented image without incorrect extra intergranules.
    """

    if len(np.unique(segmented_image)) > 2:
        raise ValueError("segmented_image must only have values of 1 and 0.")
    segmented_image_fixed = np.copy(segmented_image).astype(float)  # Float conversion for correct region labeling.
    # Add padding of IG around edges, because if edges are all GR, will ID all DM as IG
    if pad == None:
        pad = int(np.shape(segmented_image)[0]/100)
    segmented_image_fixed[:,0:pad] = 0 
    segmented_image_fixed[0:pad,:] = 0 
    segmented_image_fixed[:,-pad:] = 0 
    segmented_image_fixed[-pad:,:] = 0 
    labeled_seg = skimage.measure.label(segmented_image_fixed + 1, connectivity=2)
    values = np.unique(labeled_seg) 
    # Find value of the large continuous 0-valued region.
    size = 0
    print(f'\tloop 1 to {len(values)} (should take like 2 minutes)')
    for value in values:
        if len((labeled_seg[labeled_seg == value])) > size: # if bigger than previous largest
            if sum(segmented_image[labeled_seg == value] == 0): # if a zero (IG) region
                real_IG_value = value
                size = len(labeled_seg[labeled_seg == value])
    # Set all other 0 regions to mark value (1 or 0.5).
    print(f'\tloop 2 to {len(values)} (should take like 2 minutes)')
    for value in values:
        if np.sum(segmented_image[labeled_seg == value]) == 0:
            if value != real_IG_value:
                if not mark:
                    segmented_image_fixed[labeled_seg == value] = 1
                elif mark:
                    segmented_image_fixed[labeled_seg == value] = 0.5
    
    return segmented_image_fixed

def mark_faculae_v2b(segmented_image, data, HE_data, resolution, bp_min_flux=None, bp_max_size=0.15):
    """
    Mark faculae separately from granules - give them a value of 1.5 not 1.

    Parameters
    ----------
    segmented_image : `numpy.ndarray`
        The segmented image containing incorrect middles.
    data : `numpy array`
        The original image (not normalized or equalized)
    resolution : `float`
        Spatial resolution (arcsec/pixel) of the data.
    bp_min_flux : `float`
        Minimum flux level per pixel for a region to be considered a Bright Point.
        Default is 1 standard deviation above the mean flux (using equalized data)
    bp_max_size : `float`
        Maximum diameter (arcsec) for a region to be considered a Bright Point.
        Defualt of 0.15. 

    Returns
    -------
    segmented_image_fixed : `numpy.ndrray`
        The segmented image with faculae marked as 1.5.
    faculae_count: `int`
        The number of faculae identified in the image.
    granule_count: `int`
        The number of granules identified, after re-classifcation of faculae.
    """

    # Check inputs and initialize output map
    if len(np.unique(segmented_image)) > 3:
        raise ValueError("segmented_image must have only values of 1, 0 and a 0.5 (if dim centers marked)")
    segmented_image_fixed = np.copy(segmented_image.astype(float))
    # Set BP pixel limit
    fac_pix_limit = (bp_max_size / resolution)**2 # Max area in pixels
    # Use equalized map to set BP flux threshold and extract thresholded map
    if bp_min_flux == None: 
        bp_min_flux = np.nanmean(HE_data) + 1.25*np.nanstd(HE_data) # General flux limit determined by visual inspection.
    bright_dim_seg = np.zeros_like(data)
    bright_dim_seg[HE_data > bp_min_flux] = 1
    # Label the bright regions and get list of values
    labeled_bright_dim_seg = skimage.measure.label(bright_dim_seg + 1, connectivity=2)
    values = np.unique(labeled_bright_dim_seg)

    # Obtain gradient map and set threshold for gradient on BP edges
    grad = np.abs(np.gradient(data)[0] + np.gradient(data)[1])
    bp_min_grad = np.quantile(grad, 0.95)

    # Loop through bright regions, select those under pixel limit and containing high gradient
    fac_count = 0
    print(f'\tloop 3 to {len(values)} (this is the sticking point)')
    for value in values:
         print(f'\t\t{value}', end='\r')
         if (bright_dim_seg[labeled_bright_dim_seg==value])[0]==1: # Check region is not the non-bp region
            # check that region is small.
            region_size = len(labeled_bright_dim_seg[labeled_bright_dim_seg==value])
            if region_size < fac_pix_limit:
                # check that region has high average gradient (maybe try max gradient?)
                region_mean_grad = np.mean(grad[labeled_bright_dim_seg==value])
                if region_mean_grad > bp_min_grad:
                    segmented_image_fixed[labeled_bright_dim_seg==value] = 1.5
                    fac_count += 1
    gran_count = len(values) - 1 - fac_count  # Subtract 1 for IG region.

    return segmented_image_fixed, fac_count, gran_count

def mark_faculae_v2(segmented_image, data, resolution, bp_min_flux=None, bp_max_size=0.15):
    """
    Mark faculae separately from granules - give them a value of 1.5 not 1.

    Parameters
    ----------
    segmented_image : `numpy.ndarray`
        The segmented image containing incorrect middles.
    data : `numpy array`
        The original image (not normalized or equalized)
    resolution : `float`
        Spatial resolution (arcsec/pixel) of the data.
    bp_min_flux : `float`
        Minimum flux level per pixel for a region to be considered a Bright Point.
        Defaalt is 0.25 standard deviations above the mean flux.
    bp_max_size : `float`
        Maximum diameter (arcsec) for a region to be considered a Bright Point.
        Defualt of 0.15. 

    Returns
    -------
    segmented_image_fixed : `numpy.ndrray`
        The segmented image with faculae marked as 1.5.
    faculae_count: `int`
        The number of faculae identified in the image.
    granule_count: `int`
        The number of granules identified, after re-classifcation of faculae.
    """
    
    fac_pix_limit = (bp_max_size / resolution)**2 # Max area in pixels
    if bp_min_flux == None: 
        bp_min_flux = np.nanmean(data) + 0.25 * np.nanstd(data) # General flux limit determined by visual inspection.
    if len(np.unique(segmented_image)) > 3:
        raise ValueError("segmented_image must have only values of 1, 0 and a 0.5 (if dim centers marked)")
    segmented_image_fixed = np.copy(segmented_image.astype(float))
    labeled_seg = skimage.measure.label(segmented_image + 1, connectivity=2)

    values = np.unique(labeled_seg)
    fac_count = 0
    print(f'\tloop 3 to {len(values)} (this is the sticking point)')
    small_regions = np.zeros_like(segmented_image)
    for value in values:
        print(f'\t\t{value}', end='\r')
        mask = np.zeros_like(segmented_image)
        mask[labeled_seg == value] = 1
        # Check that is a 1 (white) region.
        if np.sum(np.multiply(mask, segmented_image)) > 0:
            region_size = len(segmented_image_fixed[mask == 1])
            # check that region is small.
            if region_size < fac_pix_limit:
                small_regions[mask == 1] = 1
                # Check that peak flux very high.
                tot_flux = np.nansum(data[mask == 1])
                if np.max(data[mask == 1]) > bp_min_flux: # if tot_flux / region_size > bp_min_flux:
                    segmented_image_fixed[mask == 1] = 1.5
                    fac_count += 1
                    small_regions[mask == 1] = 2
    gran_count = len(values) - 1 - fac_count  # Subtract 1 for IG region.

    return segmented_image_fixed, fac_count, gran_count

######################################################### 
# Algorithmic segmentation from DKISTSegmentation project 
# Some modifcations from SUNKIT-IMAGE version
#   - operate on np arrays not sunpy maps
#   - add mark_BP flag to mark bright points
######################################################### 

def segment_array(map, resolution, *, skimage_method="li", mark_dim_centers=False, mark_BP=True, fac_brightness_limit=None, fac_pix_limit=None):
    """
    IDENTICAL TO FUNCTIONS IN DKISTSegmentation REPO AND Sunkit-Image PACKAGE.
    EXCEPT RETURNING ARRAY NOT MAP AND RE-ADDEDITION OF MARK_FAC FLAG

    Segment an optical image of the solar photosphere into four-value maps with:

     * 0 as intergranule
     * 0.5 as "dim-middle"
     * 1 as granule
     * 1.5 "brightpoint"

    Parameters
    ----------
    smap : `numpy.ndarray`
        NumPy array containing data to segment.
    resolution : `float`
        Spatial resolution (arcsec/pixel) of the data.
    skimage_method : {"li", "otsu", "isodata", "mean", "minimum", "yen", "triangle"}
        scikit-image thresholding method, defaults to "li".
        Depending on input data, one or more of these methods may be
        significantly better or worse than the others. Typically, 'li', 'otsu',
        'mean', and 'isodata' are good choices, 'yen' and 'triangle' over-
        identify intergranule material, and 'minimum' over identifies granules.
    mark_dim_centers : `bool`
        Whether to mark dim granule centers as a separate category for future exploration.

    Returns
    -------
    segmented_map : `numpy.ndarray`
        NumPy array containing a segmented image (with the original header).
    """

    # if skimage_method not in METHODS:
    #     raise TypeError("Method must be one of: " + ", ".join(METHODS))

    median_filtered = sndi.median_filter(map, size=3)
    # Apply initial skimage threshold.
    threshold = get_threshold(median_filtered, skimage_method)
    segmented_image = np.uint8(median_filtered > threshold)
    # Fix the extra intergranule material bits in the middle of granules.
    seg_im_fixed = trim_intergranules(segmented_image, mark=mark_dim_centers)
    # Mark faculae and get final granule and facule count.
    if mark_BP: seg_im_markfac, faculae_count, granule_count = mark_faculae(seg_im_fixed, map, resolution)
    else: seg_im_markfac = seg_im_fixed
    # logging.info(f"Segmentation has identified {granule_count} granules and {faculae_count} faculae")
    segmented_map = seg_im_markfac
    return segmented_map

def segment(smap, resolution, *, skimage_method="li", mark_dim_centers=False):
    """
    Segment an optical image of the solar photosphere into tri-value maps with:

     * 0 as intergranule
     * 0.5 as faculae ->  NO, 1.5, RIGHT??
     * 1 as granule

    Parameters
    ----------
    smap : `~sunpy.map.GenericMap`
        `~sunpy.map.GenericMap` containing data to segment.
    resolution : `float`
        Spatial resolution (arcsec/pixel) of the data.
    skimage_method : {"li", "otsu", "isodata", "mean", "minimum", "yen", "triangle"}
        scikit-image thresholding method, defaults to "li".
        Depending on input data, one or more of these methods may be
        significantly better or worse than the others. Typically, 'li', 'otsu',
        'mean', and 'isodata' are good choices, 'yen' and 'triangle' over-
        identify intergranule material, and 'minimum' over identifies granules.
    mark_dim_centers : `bool`
        Whether to mark dim granule centers as a separate category for future exploration.

    Returns
    -------
    segmented_map : `~sunpy.map.GenericMap`
        `~sunpy.map.GenericMap` containing a segmented image (with the original header).
    """
    if not isinstance(smap, sunpy.map.mapbase.GenericMap):
        raise TypeError("Input must be an instance of a sunpy.map.GenericMap")
    if skimage_method not in METHODS:
        raise TypeError("Method must be one of: " + ", ".join(METHODS))

    median_filtered = sndi.median_filter(smap.data, size=3)
    print('median filtered')
    # Apply initial skimage threshold.
    threshold = get_threshold(median_filtered, skimage_method)
    segmented_image = np.uint8(median_filtered > threshold)
    print('applied threshold')
    # Fix the extra intergranule material bits in the middle of granules.
    seg_im_fixed = trim_intergranules(segmented_image, mark=mark_dim_centers)
    # Mark faculae and get final granule and facule count.
    seg_im_markfac, faculae_count, granule_count = mark_faculae(seg_im_fixed, smap.data, resolution)
    logging.info(f"Segmentation has identified {granule_count} granules and {faculae_count} faculae")
    segmented_map = sunpy.map.Map(seg_im_markfac, smap.meta)
    return segmented_map

def get_threshold(data, method):
    """
    Get the threshold value using given skimage segmentation type.

    Parameters
    ----------
    data : `numpy.ndarray`
        Data to threshold.
    method : {"li", "otsu", "isodata", "mean", "minimum", "yen", "triangle"}
        scikit-image thresholding method.

    Returns
    -------
    threshold : `float`
        Threshold value.
    """

    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be an instance of a np.ndarray")
    elif method == "li":
        threshold = skimage.filters.threshold_li(data)
    if method == "otsu":
        threshold = skimage.filters.threshold_otsu(data)
    elif method == "yen":
        threshold = skimage.filters.threshold_yen(data)
    elif method == "mean":
        threshold = skimage.filters.threshold_mean(data)
    elif method == "minimum":
        threshold = skimage.filters.threshold_minimum(data)
    elif method == "triangle":
        threshold = skimage.filters.threshold_triangle(data)
    elif method == "isodata":
        threshold = skimage.filters.threshold_isodata(data)
    # else:
    #     raise ValueError("Method must be one of: " + ", ".join(METHODS))
    return threshold

def trim_intergranules(segmented_image, mark=False):
    """
    Remove the erroneous identification of intergranule material in the middle
    of granules that the pure threshold segmentation produces.

    Parameters
    ----------
    segmented_image : `numpy.ndarray`
        The segmented image containing incorrect extra intergranules.
    mark : `bool`
        If `False` (the default), remove erroneous intergranules.
        If `True`, mark them as 0.5 instead (for later examination).

    Returns
    -------
    segmented_image_fixed : `numpy.ndarray`
        The segmented image without incorrect extra intergranules.
    """

    if len(np.unique(segmented_image)) > 2:
        raise ValueError("segmented_image must only have values of 1 and 0.")
    segmented_image_fixed = np.copy(segmented_image).astype(float)  # Float conversion for correct region labeling.
    labeled_seg = skimage.measure.label(segmented_image_fixed + 1, connectivity=2)
    values = np.unique(labeled_seg)
    # Find value of the large continuous 0-valued region.
    size = 0
    print(f'\tloop 1 to {len(values)} (should take like 2 minutes)')
    for value in values:
        if len((labeled_seg[labeled_seg == value])) > size: # if bigger than previous largest
            if sum(segmented_image[labeled_seg == value] == 0): # if a zero (IG) region
                real_IG_value = value
                size = len(labeled_seg[labeled_seg == value])
    # Set all other 0 regions to mark value (1 or 0.5).
    print(f'\tloop 2 to {len(values)} (should take like 2 minutes)')
    for value in values:
        if np.sum(segmented_image[labeled_seg == value]) == 0:
            if value != real_IG_value:
                if not mark:
                    segmented_image_fixed[labeled_seg == value] = 1
                elif mark:
                    segmented_image_fixed[labeled_seg == value] = 0.5
    return segmented_image_fixed

def mark_faculae(segmented_image, data, resolution):
    """
    Mark faculae separately from granules - give them a value of 1.5 not 1.

    Parameters
    ----------
    segmented_image : `numpy.ndarray`
        The segmented image containing incorrect middles.
    data : `numpy array`
        The original image.
    resolution : `float`
        Spatial resolution (arcsec/pixel) of the data.

    Returns
    -------
    segmented_image_fixed : `numpy.ndrray`
        The segmented image with faculae marked as 1.5.
    faculae_count: `int`
        The number of faculae identified in the image.
    granule_count: `int`
        The number of granules identified, after re-classifcation of faculae.
    """
    fac_size_limit = 2  # Max size of a faculae in square arcsec.
    fac_pix_limit = fac_size_limit / resolution # SHOULD SQUARE THIS???? 
    fac_brightness_limit = np.nanmean(data) + 0.5 * np.nanstd(data) # General flux limit determined by visual inspection.
    if len(np.unique(segmented_image)) > 3:
        raise ValueError("segmented_image must have only values of 1, 0 and a 0.5 (if dim centers marked)")
    segmented_image_fixed = np.copy(segmented_image.astype(float))
    labeled_seg = skimage.measure.label(segmented_image + 1, connectivity=2)
    values = np.unique(labeled_seg)
    fac_count = 0
    print(f'\tloop 3 to {len(values)} (this is the sticking point)')
    for value in values:
        print(f'\t\t{value}', end='\r')
        mask = np.zeros_like(segmented_image)
        mask[labeled_seg == value] = 1
        # Check that is a 1 (white) region.
        if np.sum(np.multiply(mask, segmented_image)) > 0:
            region_size = len(segmented_image_fixed[mask == 1])
            tot_flux = np.sum(data[mask == 1])
            # check that region is small.
            if region_size < fac_pix_limit:
                # Check that avg flux very high.
                if tot_flux / region_size > fac_brightness_limit:
                    segmented_image_fixed[mask == 1] = 1.5
                    fac_count += 1
    gran_count = len(values) - 1 - fac_count  # Subtract 1 for IG region.
    return segmented_image_fixed, fac_count, gran_count

def convert_back(seg, to='binary'):
    '''
    Convert 4-value seg into binary or tri-value (so that dont have to rerun seg algorithm)
    Both DM *and* BPs (if to=binary) are converted to GR
    '''
    outseg = np.copy(seg)
    outseg[seg == 0.5] = 1
    if to=='binary': outseg[seg == 1.5] = 1

    return outseg

def fits_to_map(filename):
    """
    Read .fits file data into a sunpy map.
    ----------
    Parameters:
        filename (string): Path to input data file (.fits format)
    ----------
    Returns:
        data_map: SunPy map containing the data and header
    """

    try:
        hdu = fits.open(filename)
        data = hdu[0].data
    except FileNotFoundError:
        raise FileNotFoundError('Cannot find ' + filename)
    except Exception:
        raise Exception('Data does not appear to be in correct .fits format')

    data_map = sunpy.map.Map(filename)

    return data_map
