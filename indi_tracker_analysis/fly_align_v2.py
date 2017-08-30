from multi_tracker_analysis import read_hdf5_file_to_pandas as mta_read

import cv2
import numpy as np
import pickle
import os

import matplotlib.pyplot as plt

import copy
import shutil

from optparse import OptionParser

SATURATION_THRESHOLD = 70
DIFFERENCE_THRESHOLD = 0.04
DO_ALIGNMENT = False
N_STD_AWAY = 3


def get_fly_roi(image_full, ellipse):
    r0 = int(ellipse[1][0])
    r1 = int(ellipse[1][1])

    r = int(np.max([r0, r1])*np.sqrt(2)/2.)

    _l = np.max([0, ellipse[0][1]-r])
    _r = np.min([image_full.shape[0], ellipse[0][1]+r])
    _width = _r - _l
    _b = np.max([0, ellipse[0][0]-r])
    _t = np.min([image_full.shape[1], ellipse[0][0]+r])
    _height = _t - _b

    zoom = copy.copy(image_full[_l:_r, _b:_t, :])
    zoom = cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB)

    return zoom

def load_fly_rois(path, color_conversion=None):
    file_list = mta_read.get_filenames(path, '.jpg')
    fly_rois = {}
    for file in file_list:
        roi = cv2.imread(file)
        if color_conversion is not None:
            roi = cv2.cvtColor(roi, color_conversion)
        fly_rois[file] = roi
    return fly_rois

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
    d = m / float(target_mean)
    roi_hsv[:,:,2] /= d
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

def get_mean_roi(directory, target_fly_filename=None, mean=np.mean):
    file_list = mta_read.get_filenames(directory, '.jpg')

    if target_fly_filename is None:
        target_fly_filename = file_list[0]

    target_fly = cv2.imread(target_fly_filename)
    f = open(os.path.join(directory, 'fly_data.pickle'))
    fly_datas = pickle.load(f) 
    f.close()

    aligned_flies = []

    for filename in file_list:
        fly = cv2.imread(filename)

        ellipse = None
        i = -1
        while ellipse is None:
            i += 1
            fly_data = fly_datas[i]
            if os.path.basename(fly_data['roi_filename']) == os.path.basename(filename):
                ellipse = fly_data['ellipse']

        aligned_fly = align_two_flies(target_fly, fly, ellipse)

        aligned_fly_hsv = cv2.cvtColor(aligned_fly, cv2.COLOR_BGR2HSV)
        aligned_flies.append(aligned_fly_hsv)

    return mean(aligned_flies, axis=0).astype(np.uint8)



def align_two_flies(im1, im2, im2_ellipse=None, conversion_to_BGR=None):
    rows,cols,channels = im2.shape

    if im2_ellipse is not None:
        ellipse = ((im2.shape[0], im2.shape[1]), (im2_ellipse[1][0], im2_ellipse[1][1]), 0)
        median_value = find_median_background_value_for_roi(im2, im2_ellipse)

    # Find size of image1
    sz = im1.shape

    if conversion_to_BGR is not None:
        im1_ = cv2.cvtColor(im1,conversion_to_BGR)
        im2_ = cv2.cvtColor(im2,conversion_to_BGR)
    else:
        im1_ = im1
        im2_ = im2

    im1_gray = cv2.cvtColor(im1_,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2_,cv2.COLOR_BGR2GRAY)

    # if the image is more white than black, invert it, because new pixels are black
    if np.median(im1_gray) > 125:
        im1_gray = 255 - im1_gray
    im1_gray -= np.min(im1_gray)

    if np.median(im2_gray) > 125:
        im2_gray = 255 - im2_gray
    im2_gray -= np.min(im2_gray)

    # Define the motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY

    # Define 3x3 matrices and initialize the matrix to identity
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 100;
    
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)


    # Use warpPerspective for Homography
    im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # set the outer circle to median background color
    if im2_ellipse is not None:
        mask_circle = np.ones_like(im2_aligned)
        r = int(im2_aligned.shape[0]/2.)
        r_buffer = int(r*2*0.2)
        zero_centered_ellipse = ((r, r), (r*2-r_buffer, r*2-r_buffer), 0) # the minus 5 deals with fringe pixels
        cv2.ellipse(mask_circle,zero_centered_ellipse,0,-1)

        im2_aligned = im2_aligned*(1-mask_circle) + (mask_circle*median_value).astype(np.uint8)

    return im2_aligned

def calc_differences_between_fly_and_models(fly, mean_fly, models, saturation_threshold=SATURATION_THRESHOLD, do_alignment=DO_ALIGNMENT):
    '''
    fly, mean_fly, and models, should all be in HSV color space
    '''
    mask = mean_fly[:,:,1]>saturation_threshold

    differences = []
    aligned_flies = []
    for model in models:
        if do_alignment:
            aligned_fly = align_two_flies(model.astype(np.uint8), fly.astype(np.uint8), conversion_to_BGR=cv2.COLOR_HSV2BGR)
        else:
            aligned_fly = fly.astype(np.uint8)
        aligned_flies.append(aligned_fly)
        fly_diff = np.mean((aligned_fly[:,:,0].astype(float) - mean_fly[:,:,0])*mask)
        model_diff = np.mean((model[:,:,0] - mean_fly[:,:,0])*mask)
        diff_diff = np.abs(fly_diff-model_diff)
        differences.append( np.mean(diff_diff) )

    return differences, aligned_flies

def calc_differences_between_fly_and_models_std(fly, mean_fly, model_directories, saturation_threshold=SATURATION_THRESHOLD):
    '''
    fly, mean_fly, and models, should all be in HSV color space
    '''
    mask = mean_fly[:,:,1]>saturation_threshold

    sat_mask = mean_fly[:,:,1]>saturation_threshold

    delta_stds = []
    model_stds = []
    for model_directory in model_directories:
        rois = load_fly_rois(model_directory, color_conversion=cv2.COLOR_BGR2HSV).values()
        s_model = np.std(rois, axis=0)[:,:,0]
        s_model = np.mean(s_model[sat_mask], axis=0)
        model_stds.append(s_model)

        rois.append(fly)
        s_with_fly = np.std(rois, axis=0)[:,:,0]
        s_with_fly = np.mean(s_with_fly[sat_mask], axis=0)

        delta_std = s_with_fly - s_model
        delta_stds.append(delta_std)

    return delta_stds, model_stds


def find_outliers_in_model(model_directory, saturation_threshold=SATURATION_THRESHOLD, N_std_away=N_STD_AWAY):
    #mean_fly = get_mean_roi(roi_directory, target_fly_filename=None, mean=np.mean)

    rois = load_fly_rois(model_directory, color_conversion=cv2.COLOR_BGR2HSV)
    if len(rois) < 3:
        return []

    aligned_rois = rois
    #target_roi = rois.values()[0]
    #for key, roi in rois.items():
    #    aligned_roi = roi #align_two_flies(target_roi, roi, im1_ellipse=None, conversion_to_BGR=cv2.COLOR_HSV2BGR)
    #    aligned_rois[key] = aligned_roi

    mean_model = np.mean(aligned_rois.values(), axis=0)
    sat_mask = mean_model[:,:,1]>saturation_threshold

    stds = {}

    for key_to_exclude in aligned_rois.keys():
        keys_to_include = [k for k in aligned_rois.keys() if k != key_to_exclude]
        s = np.std([aligned_rois[k] for k in keys_to_include], axis=0)[:,:,0]
        s = np.mean(s[sat_mask], axis=0)
        stds[key_to_exclude] = s

    mean_stds = np.mean(stds.values())
    std_stds = np.std(stds.values())

    outliers = []

    for key, std in stds.items():
        if np.abs(std - mean_stds) > std_stds*N_std_away:
            outliers.append(key)

    return outliers

def calc_stds_for_model(model_directory, saturation_threshold=SATURATION_THRESHOLD):
    #mean_fly = get_mean_roi(roi_directory, target_fly_filename=None, mean=np.mean)

    rois = load_fly_rois(model_directory, color_conversion=cv2.COLOR_BGR2HSV)
    if len(rois) < 3:
        return

    aligned_rois = rois
    #target_roi = rois.values()[0]
    #for key, roi in rois.items():
    #    aligned_roi = roi #align_two_flies(target_roi, roi, im1_ellipse=None, conversion_to_BGR=cv2.COLOR_HSV2BGR)
    #    aligned_rois[key] = aligned_roi

    mean_model = np.mean(aligned_rois.values(), axis=0)
    sat_mask = mean_model[:,:,1]>saturation_threshold

    s = np.std(aligned_rois.values(), axis=0)[:,:,0]
    s = np.mean(s[sat_mask], axis=0)

    print s

def remove_outliers_from_model(model_directory, outlier_directory=None):
    if outlier_directory is None:
        outlier_directory = os.path.join(os.path.dirname(model_directory), 'model_outliers')
        if os.path.exists(outlier_directory):
            pass
        else:
            os.mkdir(outlier_directory)

        outliers = find_outliers_in_model(model_directory, saturation_threshold=SATURATION_THRESHOLD, N_std_away=N_STD_AWAY)

        if len(outliers) > 0:
            print 'Moving the following files to the outlier directory: '
            for outlier in outliers:
                print outlier
            print

        for outlier in outliers:
            new_location = os.path.join(outlier_directory, os.path.basename(outlier) )
            shutil.move(outlier, new_location)



def get_mean_roi_from_aligned_directory_in_hsv(directory, do_alignment=False):
    rois = load_fly_rois(directory, color_conversion=cv2.COLOR_BGR2HSV)
    if do_alignment:
        target_roi = rois.values()[0]
        for key, roi in rois.items():
            aligned_roi = align_two_flies(target_roi, roi, im2_ellipse=None, conversion_to_BGR=cv2.COLOR_HSV2BGR)
            rois[key] = aligned_roi
    return np.mean(rois.values(), axis=0)

def find_best_model_for_outliers(roi_directory, difference_threshold=DIFFERENCE_THRESHOLD):
    outlier_directory = mta_read.get_filename(roi_directory, 'model_outliers')
    outlier_filenames = mta_read.get_filenames(outlier_directory, '.jpg')
    for outlier_filename in outlier_filenames:
        assign_fly_to_model_and_update_models_from_filenames(roi_directory, outlier_filename, difference_threshold=difference_threshold)
        os.remove(outlier_filename)

def remove_outliers_and_update_all_models(roi_directory):
    model_directories = mta_read.get_filenames(roi_directory, 'model_', ['outlier'])
    for model_directory in model_directories:
        remove_outliers_from_model(model_directory, outlier_directory=None)
    find_best_model_for_outliers(roi_directory)

def assign_fly_to_model(fly, mean_fly, model_directories, difference_threshold=DIFFERENCE_THRESHOLD):
    '''
    fly: roi
    mean_fly: mean of all rois
    models: means of model groups
    everything in hsv, and pre=aligned

    '''
    if len(model_directories)>0:
        models = []
        for model_directory in model_directories:
            model = get_mean_roi_from_aligned_directory_in_hsv(model_directory, do_alignment=DO_ALIGNMENT)
            models.append(model)
    else:
        models = []

    differences, aligned_flies = calc_differences_between_fly_and_models(   fly, 
                                                                            mean_fly, 
                                                                            models, 
                                                                            saturation_threshold=SATURATION_THRESHOLD)
    print differences
    if np.min(differences) < difference_threshold:
        idx = np.argmin(differences)
        return idx, aligned_flies[idx]
    else:
        return None

def assign_fly_to_model_std(fly, mean_fly, model_directories, difference_threshold=DIFFERENCE_THRESHOLD):
    '''
    fly: roi
    mean_fly: mean of all rois
    models: means of model groups
    everything in hsv, and pre=aligned

    '''

    delta_stds, model_stds = calc_differences_between_fly_and_models_std(   fly, 
                                                                            mean_fly, 
                                                                            model_directories, 
                                                                            saturation_threshold=SATURATION_THRESHOLD)
    print delta_stds
    print 'target minimum: ', np.std(model_stds)*N_STD_AWAY
    if np.min(delta_stds) < np.std(model_stds)*N_STD_AWAY:
        idx = np.argmin(delta_stds)
        return idx, fly
    else:
        return None

def assign_fly_to_model_and_update_models(roi_directory, fly_filename, fly, mean_fly, model_directories, difference_threshold=DIFFERENCE_THRESHOLD):        
    if len(model_directories) > 0:
        number_of_flies_per_model = [len(mta_read.get_filenames(model_directory, '.jpg')) for model_directory in model_directories]
        for i, N in enumerate(number_of_flies_per_model):
            if N >= 3:
                remove_outliers_from_model(model_directories[i], outlier_directory=None)
        if np.min(number_of_flies_per_model) >= 2:
            best_model_fit = assign_fly_to_model_std(fly, mean_fly, model_directories, difference_threshold)
            print 'std method' 
            #

        else:
            best_model_fit = assign_fly_to_model(fly, mean_fly, model_directories, difference_threshold)
    else:
        best_model_fit = None

    if best_model_fit is None: # make new model
        new_model_number = len(model_directories)
        model_directory = os.path.join(roi_directory, 'model_'+str(new_model_number))
        if os.path.exists(model_directory):
            pass
        else:
            os.mkdir(model_directory)
        aligned_fly = fly
    else:
        idx, aligned_fly = best_model_fit
        model_directory = os.path.join(roi_directory, 'model_'+str(idx))

    aligned_fly_bgr = cv2.cvtColor(aligned_fly, cv2.COLOR_HSV2BGR)
    f = os.path.join(model_directory, os.path.basename(fly_filename))
    #cv2.imwrite(f, aligned_fly_bgr) 
    shutil.copyfile(fly_filename, f)
    if 'outlier' in fly_filename:
        os.remove(fly_filename)

def assign_fly_to_model_and_update_models_from_filenames(roi_directory, fly_filename, difference_threshold=DIFFERENCE_THRESHOLD):
    model_directories = mta_read.get_filenames(roi_directory, 'model_', ['outlier'])



    mean_fly = get_mean_roi_from_aligned_directory_in_hsv(roi_directory)

    fly = cv2.imread(fly_filename)
    fly = cv2.cvtColor(fly, cv2.COLOR_BGR2HSV)

    assign_fly_to_model_and_update_models(roi_directory, fly_filename, fly, mean_fly, model_directories, difference_threshold=difference_threshold)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--path", type="str", dest="path", default='',
                        help="directory where images can be found")

    (options, args) = parser.parse_args()
    
    roi_filenames =  mta_read.get_filenames(options.path, '.jpg')
    while len(roi_filenames) > 0:
        roi_filename = roi_filenames.pop(0)
        assign_fly_to_model_and_update_models_from_filenames(options.path, roi_filename, difference_threshold=DIFFERENCE_THRESHOLD  )

        outlier_directory = os.path.join(options.path, 'model_outliers')
        if os.path.exists(outlier_directory):
            outlier_filenames = mta_read.get_filenames(outlier_directory, '.jpg')
            roi_filenames.extend(outlier_filenames)


    #remove_outliers_and_update_all_models(options.path)