import cv2
import numpy as np


SATURATION_THRESHOLD = 70
#DIFFERENCE_THRESHOLD = 0.04
#DO_ALIGNMENT = False
#N_STD_AWAY = 3

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

def get_saturation_mask(roi, threshold=SATURATION_THRESHOLD):
    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    mask = hsv[:,:,1]>threshold
    return mask

def rotate_image_by_180_to_align_saturation_values(roi, threshold=SATURATION_THRESHOLD):
    sat_mask = get_saturation_mask(roi, threshold=threshold)
    if np.mean(sat_mask[int(roi.shape[0]/2.):,:]) > np.mean(sat_mask[0:int(roi.shape[0]/2.),:]):
        roi = rotate_image_by_180(roi)
    return roi

def rough_align_fly_in_roi_to_vertical_position(roi, ellipse):
    roi = rotate_roi(roi, ellipse)
    roi = rotate_image_by_180_to_align_saturation_values(roi)
    return roi

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