import numpy as np
import cv2
import sunpy
import scipy.ndimage as sndi
import skimage
import pandas as pd
from sklearn import preprocessing

'''
Functions for the machine learning segmentation of various solar features
'''

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

def post_process(preds):
    if len(np.unique(preds)) == 2:
        preds = np.copy(preds).astype(float)  # Float conversion for correct region labeling.
        preds2 = np.ones_like(preds)*20 # Just to aviod issues later on 
        labeled_preds = skimage.measure.label(preds + 1, connectivity=2)
        values = np.unique(labeled_preds)
        # Find value of the largest region (IG region)
        size = 0
        for value in values:
            if len((labeled_preds[labeled_preds == value])) > size:
                IG_value = value
                size = len(labeled_preds[labeled_preds == value])
        # Where labeled_preds=IG_value set preds2 to zero, otherwise 1
        preds2[labeled_preds == IG_value] = 0
        preds2[labeled_preds != IG_value] = 1  
    else: print('NOT YET IMPLEMENTED: NEED TO FIND A WAY TO GET 3-VALUE SEG INTO FORM COMPARABLE TO LABELS'); a=b
    # uniques = np.unique(preds)
    # preds2 = np.copy(preds)
    # if  len(uniques) == 2:
    #     # assign the continuous region to be 0s
    #     flat = preds.reshape(-1)
    #     if len(np.where(flat == uniques[0])[0]) > len(np.where(flat == uniques[1])[0]):
    #         preds2[np.where(preds==uniques[0])] == 0
    #         preds2[np.where(preds==uniques[1])] == 1
    #     elif len(np.where(flat == uniques[0])[0]) < len(np.where(flat == uniques[1])[0]):
    #         preds2[np.where(preds==uniques[0])] == 1
    #         preds2[np.where(preds==uniques[1])] == 0
    #     else: print(''); a=b
    # else: 
    #     print('NOT YET IMPLEMENTED: NEED TO FIND A WAY TO GET 3-VALUE SEG INTO FORM CMPARABLE TO LABELS')
    #     a=b
    return preds2


######## Functions from DKISTSegmentation project for validation of ML methods ###########

def segment_array(map, resolution, *, skimage_method="li", mark_dim_centers=False, mark_BP=True):
    """
    SIMILAR BUT NOT IDENTICAL TO FUNCTIONS IN DKISTSegmentation REPO AND Sunkit-Image PACKAGE, BUT 
    WITH SMALL MODIFICATIONS, E.G.,RE-ADDITION OF MARK_FAC FLAG

    Segment an optical image of the solar photosphere into tri-value maps with:

     * 0 as intergranule
     * 0.5 as faculae
     * 1 as granule

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
     * 0.5 as faculae
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
    # Apply initial skimage threshold.
    threshold = get_threshold(median_filtered, skimage_method)
    segmented_image = np.uint8(median_filtered > threshold)
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
    labeled_seg = skimage.measure.label(segmented_image + 1, connectivity=2)
    values = np.unique(labeled_seg)
    # Find value of the large continuous 0-valued region.
    size = 0
    for value in values:
        if len((labeled_seg[labeled_seg == value])) > size:
            real_IG_value = value
            size = len(labeled_seg[labeled_seg == value])
    # Set all other 0 regions to mark value (1 or 0.5).
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
    Mark faculae separately from granules - give them a value of 2 not 1.

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
    fac_pix_limit = fac_size_limit / resolution
    # General flux limit determined by visual inspection.
    fac_brightness_limit = np.mean(data) + 0.5 * np.std(data)
    if len(np.unique(segmented_image)) > 3:
        raise ValueError("segmented_image must have only values of 1, 0 and a 0.5 (if dim centers marked)")
    segmented_image_fixed = np.copy(segmented_image.astype(float))
    labeled_seg = skimage.measure.label(segmented_image + 1, connectivity=2)
    values = np.unique(labeled_seg)
    fac_count = 0
    for value in values:
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