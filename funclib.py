import numpy as np
import cv2
import sunpy
import scipy.ndimage as sndi
import skimage
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
import astropy.io.fits as fits 
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import skimage as sk

######## Functions for NNs

class multiclass_MSE_loss(nn.Module):

    def __init__(self):
        super(multiclass_MSE_loss, self).__init__()

    def forward(self, preds, targets, dm_weight=1, bp_weight=1):
        """
        Compute MSE between preds and targets for each layer (Ig, DM, ..etc). 
        Sum to get total, applying wieghting to small classes.
        NOTE: in this blog, they do a similar thing for image classification https://towardsdatascience.com/implementing-custom-loss-functions-in-pytorch-50739f9e0ee1

        Parameters
        ----------
        preds : `numpy.ndarray` of shape [n_obs, n_classes, n_pix, n_pix]
            Model outputs before application of activation function
        
        Returns
        -------
        targets : `numpy.ndarray` [n_obs, n_classes, n_pix, n_pix]
            One-hot encoded target values. Ordering along n_class axis must be the same as for preds.
        """

        preds = torch.sigmoid(preds) # use sigmiod to turn into class probs
        mse_ig, mse_dm, mse_gr, mse_bp = 0, 0, 0, 0
        mse = nn.MSELoss()
        for idx in range(preds.shape[0]): # loop through images in batch
            mse_ig += mse(targets[idx,0,:,:], preds[idx,0,:,:])
            mse_dm += mse(targets[idx,1,:,:], preds[idx,1,:,:]) * dm_weight
            mse_gr += mse(targets[idx,2,:,:], preds[idx,2,:,:])
            mse_bp += mse(targets[idx,3,:,:], preds[idx,3,:,:]) * bp_weight
        loss =  mse_ig + dm_weight*mse_dm + mse_gr + bp_weight*mse_bp
        return loss

def check_inputs(train_ds, train_loader):

    # Check data is loaded correctly
    print('Train data:')
    print(f'     {len(train_ds)} obs, broken into {len(train_loader)} batches')
    train_features, train_labels = next(iter(train_loader))
    shape = train_features.size()
    print(f'     Each batch has data of shape {train_features.size()}, e.g. {shape[0]} images, {[shape[2], shape[3]]} pixels each, {shape[1]} layers (features)')
    shape = train_labels.size()
    print(f'     Each batch has labels of shape {train_labels.size()}, e.g. {shape[0]} images, {[shape[2], shape[3]]} pixels each, {shape[1]} layers (classes)')

def get_feature(img, name, index):

    if name == 'gradx': a = np.gradient(img)[0]
    elif name == 'grady': a = np.gradient(img)[1]
    elif name == 'squared': a = img**2
    elif 'power' in name:
        n = int(name[-1])
        a = img**n
    # elif name == 'deltaBinImg':
    #     UNet1seg = np.load(f'../UNet1_outputs/pred_{index}.npy') # zeros and ones
    #     imgnorm = (img - np.mean(img))/np.std(img) # normalize to range [0, 1]
    #     a = UNet1seg - imgnorm # difference between binary segmentation and image (pos for bp, neg for dm)
    else: raise ValueError(f'Channel name {name} not recognized')

    return a

######## Preprocessing 

def histogram_equalization(data, n_bins=256):

   #get image histogram
   hist, bins = np.histogram(data.flatten(), n_bins, normed=True)
   cdf = hist.cumsum()
   cdf = 255 * cdf / cdf[-1] #normalize
   # use linear interpolation of cdf to find new pixel values
   data_out = np.interp(data.flatten(), bins[:-1], cdf)
   data_out = data_out.reshape(data.shape)

   return data_out

def get_localRMS(data):
    # Should I do this?
    return None

def match_to_firstlight(obs_data, n_bins=2000):

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

######## Functions for the machine learning segmentation of various solar features

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
        # fig, (ax1, ax2) = plt.subplots(1,2); ax1.imshow(data); ax2.imshow(filtered_img)
        df_new['kernel'+str(i)] = filtered_img
    return df_new

def add_gradient_feats(df, data):

    # Attempted to use cv2.Laplacian, but require dtype uint8, and converting caused issues with the normalization #cv2.Laplacian(data,cv2.CV_64F).reshape(-1)
    df_new = df.copy()
    df_new['gradienty'] = np.gradient(data)[0].reshape(-1)
    df_new['gradientx'] = np.gradient(data)[1].reshape(-1)
    return df_new

def add_sharpening_feats(df, dataflat):

    df_new = df.copy()
    df_new['value2'] = dataflat**2
    return df_new

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

def post_process(preds, data):

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

def eval_metrics(metric, true, pred):
    """
    Ways of evaluating extent to which true and preds are the same, BUT ONLY IN THE CASE THAT THEY SHOULD BE.
    E.g. for comparing labels to outputs of surpervised method, or to outputs of 2-value unsupervised method
    where those two values have been converted to 0 = IG, 1 = G.
    None of these will be usefull for >2-value unsupervised methods untill I figure out how to algorithmically
    force the outputs (including combining groups) into IG, G, BP, DM values. 
    """
    if metric == 'pct_correct':
        # Could be ok for our use; in gernal bad if huge class imbalance (could get high score by predicting all one class)
        return len(np.where(preds==true)[0])/len(preds)
    if metric == 'accuracy_score':
        # Avg of the area of overlap over area of union for each class (like Jaccard score but for two or more classes)
        return metrics.accuracy_score(true, pred)

######## Functions for use with UNet tutorial 2

import torch
import torchvision

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval() # set model into eval mode
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"saved_images/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
    model.train() # set model back into train mode


######## Functions from DKISTSegmentation project for validation of ML methods ###########

def segment_array_v2(map, resolution, *, skimage_method="li", mark_dim_centers=False, mark_BP=True, bp_min_flux=None, bp_max_size=0.15):
    
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
        Defaalt is half a standard deviation above the mean flux.
    bp_max_size: `float`
        Maximum diameter (arcsec) to consider a region a Bright Point.
        Defualt of 0.15. 

    Returns
    -------
    segmented_map : `numpy.ndarray`
        NumPy array containing a segmented image (with the original header).
    """

    # if skimage_method not in METHODS:
    #     raise TypeError("Method must be one of: " + ", ".join(METHODS))

    # Perform local histogram equalization on map.
    map_norm = ((map - np.nanmin(map))/(np.nanmax(map) - np.nanmin(map))) * 225 # min-max normalization to [0, 225] 
    map_HE = sk.filters.rank.equalize(map_norm.astype(int), footprint=sk.morphology.disk(250)) # MAKE FOOTPRINT SIZE DEPEND ON RESOLUTION!!!
    # Apply initial skimage threshold.
    median_filtered = sndi.median_filter(map_HE, size=3)
    threshold = get_threshold_v2(median_filtered, skimage_method)
    segmented_image = np.uint8(median_filtered > threshold)
    # Fix the extra intergranule material bits in the middle of granules.
    seg_im_fixed = trim_intergranules_v2(segmented_image, mark=mark_dim_centers)
    # Mark faculae and get final granule and facule count.
    if mark_BP: seg_im_markfac, faculae_count, granule_count = mark_faculae_v2(seg_im_fixed, map, resolution, bp_min_flux, bp_max_size)
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
    # else:
    #     raise ValueError("Method must be one of: " + ", ".join(METHODS))
    return threshold

def trim_intergranules_v2(segmented_image, mark=False):
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
    # Add padding of IG around edges, because if edges are all GR, will ID all DM as IG (should not affect final output)
    segmented_image[:,0:20] = 0 
    segmented_image[0:20,:] = 0 
    segmented_image[:,-20:] = 0 
    segmented_image[-20:,:] = 0 
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

    # fig, ((ax1, ax2, ax3),(ax4, ax5, ax6))= plt.subplots(2, 3, figsize=(15, 10))
    # im = ax1.imshow(segmented_image[2400:2550, 2150:2450], origin='lower')
    # im = ax2.imshow(data[2400:2550, 2150:2450], origin='lower')
    # im = ax3.imshow(labeled_seg[2400:2550, 2150:2450], origin='lower')
    # ax3.set_title(np.unique(labeled_seg[2400:2550, 2150:2450]))
    # im = ax4.imshow(segmented_image[2100:2300, 2100:2500], origin='lower')
    # im = ax5.imshow(data[2100:2300, 2100:2500], origin='lower')
    # im = ax6.imshow(labeled_seg[2100:2300, 2100:2500], origin='lower')
    # ax6.set_title(np.unique(labeled_seg[2100:2300, 2100:2500]))
    # plt.savefig('test0626')
    # a=b
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

    # im = plt.imshow(small_regions, origin='lower'); plt.colorbar(im)
    # plt.title('small regions (1 = bright enough)')
    # plt.savefig('test0625b')
    # # plt.hist(data.flatten(), bins=20, label='all pixels')
    # # plt.hist(all_region_maxfluxes, label='all region maximums')
    # # plt.hist(small_region_maxfluxes, label='small region maximums')
    # # plt.legend()
    # # plt.vlines([np.nanmean(data)+0.25*np.nanstd(data)], ymin=0, ymax=7e6, color='black', label='mean+0.25*SD')
    # # plt.legend()
    # # plt.yscale('log')
    # # plt.title('VBI_04_05_4096')
    # # plt.savefig('test0625')

    return segmented_image_fixed, fac_count, gran_count


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



    # #### PLACE THIS CODE WITHIN TRIM_INTERGRGRANULES FUNCTION ###
    # print(real_IG_value)
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(50, 10))
    # im = ax1.imshow(segmented_image[0:2000, 0:2000], origin='lower')
    # plt.colorbar(im, ax=ax1)
    # im = ax2.imshow(labeled_seg[0:2000, 0:2000], origin='lower')
    # plt.colorbar(im, ax=ax2); ax2.set_title('label values')
    # region1, region2, region3 = np.zeros_like(labeled_seg)*np.NaN, np.zeros_like(labeled_seg)*np.NaN, np.zeros_like(labeled_seg)*np.NaN
    # region1[labeled_seg==values[0]] = values[0]; print(np.where(labeled_seg==values[0]))
    # region2[labeled_seg==values[1]] = values[1]
    # region3[labeled_seg==values[2]] = values[2]; region3[labeled_seg==values[1]] = values[1] 
    # im = ax3.imshow(region1[0:2000, 0:2000], origin='lower')
    # plt.colorbar(im, ax=ax3); ax3.set_title('value 1')
    # im = ax4.imshow(region2[0:2000, 0:2000], origin='lower')
    # plt.colorbar(im, ax=ax4); ax4.set_title('value 2')
    # im = ax5.imshow(region3[0:2000, 0:2000], origin='lower')
    # plt.colorbar(im, ax=ax5); ax5.set_title('value 3 and 2')
    # plt.savefig('test0622')
    # a=b
    ###################