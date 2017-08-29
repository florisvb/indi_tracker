from multi_tracker_analysis import read_hdf5_file_to_pandas as mta_read

import cv2
import numpy as np
import pickle
import os

import matplotlib.pyplot as plt

import copy
from optparse import OptionParser


def get_fly_roi(image_full, ellipse):
    r0 = int(ellipse[1][0])
    r1 = int(ellipse[1][1])

    r = int(np.max([r0, r1]))*np.sqrt(2)/2.

    _l = np.max([0, ellipse[0][1]-r])
    _r = np.min([image_full.shape[0], ellipse[0][1]+r])
    _width = _r - _l
    _b = np.max([0, ellipse[0][0]-r])
    _t = np.min([image_full.shape[1], ellipse[0][0]+r])
    _height = _t - _b

    zoom = copy.copy(image_full[_l:_r, _b:_t, :])
    zoom = cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB)

    return zoom

def get_elliptical_fly_mask(roi, ellipse, ellipse_value=1):
    r = roi.shape[0]/2. # assumes square roi
    mask_ellipse = np.ones_like(roi)
    zero_centered_ellipse = ((r, r), (ellipse[1][0], ellipse[1][1]), ellipse[2])
    cv2.ellipse(mask_ellipse,zero_centered_ellipse,0,-1)
    if ellipse_value == 1:
        return 1 - mask_ellipse
    else:
        return mask_ellipse

def find_median_background_value_for_roi(roi, ellipse):
    mask_ellipse = get_elliptical_fly_mask(roi, ellipse, ellipse_value=0)
    masked = roi*mask_ellipse
    return np.median(np.median(masked, axis=0), axis=0)

def get_saturation_mask(roi, threshold=60):
    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    mask = hsv[:,:,1]>threshold
    return mask

def rotate_image_by_180_to_align_saturation_values(roi, threshold=120):
    sat_mask = get_saturation_mask(roi, threshold=threshold)
    if np.mean(sat_mask[int(roi.shape[0]/2.):,:]) > np.mean(sat_mask[0:int(roi.shape[0]/2.),:]):
        roi = rotate_image_by_180(roi)
    return roi

def rough_align_fly_in_roi_to_vertical_position(roi, ellipse):
    roi = rotate_roi(roi, ellipse)
    roi = rotate_image_by_180_to_align_saturation_values(roi)
    return roi

def balance_luminance(roi, ellipse, target_mean=40):
    # adjust overall luminance so that flies are all same rough luminance
    mask_ellipse = get_elliptical_fly_mask(roi, ellipse, ellipse_value=1)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(float)
    m = np.mean(roi_hsv[:,:,2]*mask_ellipse[:,:,2])
    print m
    d = m - float(target_mean)
    roi_hsv[:,:,2] -= d
    print m, float(target_mean), d, 'new mean: ', np.mean(roi_hsv[:,:,2]*mask_ellipse[:,:,2])
    np.clip(roi_hsv[:,:,2], 0, 255, out=roi_hsv[:,:,2])
    roi_bgr = cv2.cvtColor(roi_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return roi_bgr

def rotate_image_by_180(roi):
    rows,cols,channels = roi.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
    roi = cv2.warpAffine(roi,M,(cols,rows))
    return roi

def rotate_roi(roi, ellipse):
    '''
    Rotations introduce black pixels in places where the original image does not provide coverage.
    We replace the outer circle with the median of the images outside of the fly ellipse to remove those black pixels.
    '''
    median_value = find_median_background_value_for_roi(roi, ellipse)

    rows,cols,channels = roi.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),ellipse[2],1)
    roi = cv2.warpAffine(roi,M,(cols,rows))

    r = roi.shape[0]/2.
    mask_circle = np.ones_like(roi)
    zero_centered_ellipse = ((r, r), (r*2-5, r*2-5), 0) # the minus 5 deals with fringe pixels
    cv2.ellipse(mask_circle,zero_centered_ellipse,0,-1)

    roi = roi*(1-mask_circle) + (mask_circle*median_value).astype(np.uint8)

    return roi

def get_median_roi(directory, target_fly_filename=None):
    file_list = mta_read.get_filenames(directory, '.jpg')

    if target_fly_filename is None:
        target_fly_filename = file_list[0]

    target_fly = cv2.imread(target_fly_filename)

    aligned_flies = []

    for filename in file_list:
        fly = cv2.imread(filename)
        aligned_fly = align_two_flies(target_fly, fly)

        aligned_fly_hsv = cv2.cvtColor(aligned_fly, cv2.COLOR_BGR2HSV)
        aligned_flies.append(aligned_fly_hsv)

    return np.median(aligned_flies, axis=0).astype(np.uint8)



def align_two_flies(im1, im2):
    rows,cols,channels = im2.shape

    # Find size of image1
    sz = im1.shape

    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

    # if the image is more white than black, invert it, because new pixels are black
    if np.median(im1_gray) > 125:
        im1_gray = 255 - im1_gray
    im1_gray -= np.min(im1_gray)

    if np.median(im2_gray) > 125:
        im2_gray = 255 - im2_gray
    im2_gray -= np.min(im2_gray)

    # Define the motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 100;
    
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)


    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    print warp_matrix

    return im2_aligned

