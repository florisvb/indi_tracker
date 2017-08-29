from multi_tracker_analysis import read_hdf5_file_to_pandas as mta_read

import cv2
import numpy as np
import pickle
import os

import matplotlib.pyplot as plt

import copy
from optparse import OptionParser

from distutils.version import LooseVersion, StrictVersion
print 'Using open cv: ' + cv2.__version__
if StrictVersion(cv2.__version__.split('-')[0]) >= StrictVersion("3.0.0"):
    OPENCV_VERSION = 3
    print 'Open CV 3'
else:
    OPENCV_VERSION = 2
    print 'Open CV 2'

def create_median_gray_small_image_from_directory(directory, N=3, resize_factor=0.2):
    # N is the number of equidistant files to use for making the median
    file_list = mta_read.get_filenames(directory, '.jpg')
    imgs = []
    indices = np.linspace(0,len(file_list)-1,N).astype(int)
    for i in indices:
        file = file_list[i]
        img = cv2.imread(file, cv2.CV_8UC1)
        small = cv2.resize(img, (0,0), fx=resize_factor, fy=resize_factor) 
        imgs.append(small)
    median = np.median(imgs, axis=0)
    return median.astype(np.uint8)

def find_fly_in_image(image, median, threshold=40, pixels_per_mm=10, min_fly_length_mm=1, max_fly_ecc=5):
    '''
    pixels_per_mm - after resize transformation
    '''
    absdiff = cv2.absdiff(image, median)
    retval, threshed = cv2.threshold(absdiff, threshold, 255, 0)
    
    kern_d = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_d,kern_d))
    threshed = cv2.morphologyEx(threshed,cv2.MORPH_OPEN, kernel, iterations = 1)
    
    kernel = np.ones((3,3), np.uint8)
    threshed = cv2.dilate(threshed, kernel, iterations=5)
    threshed = cv2.erode(threshed, kernel, iterations=2)

    # http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
    if OPENCV_VERSION == 2:
        contours, hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elif OPENCV_VERSION == 3:
        threshed = np.uint8(threshed)
        image, contours, hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fly_ellipses = []
    for contour in contours:
        if len(contour) > 5:
            ellipse = cv2.fitEllipse(contour)
            fly_length = np.max(ellipse[1])
            fly_width = np.min(ellipse[1])
            fly_ecc = fly_length / fly_width
            if fly_length > min_fly_length_mm and fly_ecc < max_fly_ecc:
                fly_ellipses.append(ellipse)
    
    return fly_ellipses


def find_flies_in_images(directory, 
                         resize_factor=0.2, 
                         threshold=40, 
                         pixels_per_mm=10, 
                         min_fly_length_mm=1, 
                         max_fly_ecc=5,
                         save_result=True):
    
    try:
        config = read_hdf5_file_to_pandas.load_config_from_path(path)
        s = config.identifiercode + '_' + 'gphoto2'
        gphoto2directory = os.path.join(config.path, s)
        directory = gphoto2directory
    except:
        pass

    median = create_median_gray_small_image_from_directory(directory,  
                                                           N=3, 
                                                           resize_factor=resize_factor)
    
    flies = {}
    file_list = mta_read.get_filenames(directory, '.jpg')
    for file in file_list:
        img = cv2.imread(file, cv2.CV_8UC1)
        small = cv2.resize(img, (0,0), fx=resize_factor, fy=resize_factor) 
    
        fly_ellipses = find_fly_in_image(small, 
                                         median, 
                                         threshold=threshold, 
                                         pixels_per_mm=pixels_per_mm, 
                                         min_fly_length_mm=min_fly_length_mm, 
                                         max_fly_ecc=max_fly_ecc)
        
        large_fly_ellipses = []
        for ellipse in fly_ellipses:
            large_ellipse = ((ellipse[0][0]/resize_factor, ellipse[0][1]/resize_factor), 
                             (ellipse[1][0]/resize_factor, ellipse[1][1]/resize_factor), 
                             ellipse[2])
            large_fly_ellipses.append(large_ellipse)
        
        flies[os.path.basename(file)] = large_fly_ellipses
        
    if save_result:
        fname = os.path.join(directory, 'fly_ellipses.pickle')
        f = open(fname, 'w')
        pickle.dump(flies, f)
        f.close()

def show_fly_ellipse(path, filebasename):
    filebasename = os.path.basename(filebasename)
    file = os.path.join(path, filebasename)
    img = cv2.imread(file, cv2.CV_8UC1)

    fname = os.path.join(path, 'fly_ellipses_and_colors.pickle')
    if not os.path.exists(fname):
        raise ValueError('Please run find_flies_in_image_directory.find_flies_in_directory')
    f = open(fname)
    gphoto2_flies = pickle.load(f)
    f.close()

    for i in range(len(gphoto2_flies[filebasename])):
        img = cv2.ellipse(img,gphoto2_flies[filebasename][i]['ellipse'],color=(0,255,0),thickness=20)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(img)

def calculate_hue_for_flies(path):
    try:    
        config = read_hdf5_file_to_pandas.load_config_from_path(path)
        s = config.identifiercode + '_' + 'gphoto2'
        gphoto2directory = os.path.join(config.path, s)
    except:
        gphoto2directory = path

    fname = os.path.join(gphoto2directory, 'fly_ellipses.pickle')
    if not os.path.exists(fname):
        raise ValueError('Please run find_flies_in_image_directory.find_flies_in_directory')
    f = open(fname)
    gphoto2_flies = pickle.load(f)
    f.close()

    fly_ellipses_and_colors = {}

    all_hues = []

    for filename, flies in gphoto2_flies.items():
        gphoto2img = cv2.imread(os.path.join(gphoto2directory, filename) )
        fly_ellipses_and_colors[os.path.basename(filename)] = []
        for i, ellipse in enumerate(flies):
            r0 = int(ellipse[1][0])
            r1 = int(ellipse[1][1])

            _l = np.max([0, ellipse[0][1]-r1])
            _r = np.min([gphoto2img.shape[0], ellipse[0][1]+r1])
            _width = _r - _l
            _b = np.max([0, ellipse[0][0]-r0])
            _t = np.min([gphoto2img.shape[1], ellipse[0][0]+r0])
            _height = _t - _b
            zoom = gphoto2img[int(_l):int(_r), int(_b):int(_t), :]


            zoom_hsv = cv2.cvtColor(zoom, cv2.COLOR_BGR2HSV)
            # http://docs.opencv.org/trunk/df/d9d/tutorial_py_colorspaces.html

            # ellipse mask
            mask_ellipse = copy.copy(zoom[:,:,0])*0
            zero_centered_ellipse = (( np.max([0,zoom.shape[1]-r0]), np.max([0,zoom.shape[0]-r1])), (ellipse[1][0], ellipse[1][1]), ellipse[2])
            cv2.ellipse(mask_ellipse,zero_centered_ellipse,1,-1)

            # saturation mask
            mask_sat = copy.copy(zoom_hsv[:,:,1])
            mask_sat = mask_sat>10

            # luminance mask
            mask_lum = copy.copy(zoom_hsv[:,:,2])
            mask_lum = (mask_lum>10)*(mask_lum<240)

            mask = mask_ellipse#*mask_sat*mask_lum
            mask /= np.max(mask)




            fly = {'ellipse': ellipse,
                   'luminance': np.mean(zoom_hsv[:,:,2]*mask),
                   #'hue_actual': hist_hue,
                   #'hue_bins': bins,
                   #'rgb_color_peak': rgb_color.tolist()}
                }

            fly_ellipses_and_colors[os.path.basename(filename)].append(fly)

    # get hue relative to the baseline. useful for focusing on small differences relative to a common baseline
    # could try to cluster first, then get baseline based on selecting even number of flies from each cluster
    baseline_hue = np.mean(all_hues, axis=0).astype('float')

    if 0:
        for filename, flies in fly_ellipses_and_colors.items():
            for i, fly in enumerate(flies):
                pass
                #hue = fly['hue_actual'].astype(float)
                #hue_relative = hue - baseline_hue
                #hue_relative -= np.min(hue_relative)
                #hue_relative /= float(np.sum(hue_relative))
                #fly['hue_relative'] = hue_relative

                #primary_hue = fly['hue_bins'][np.argmax(hue_relative)]
                #hsv_color = np.uint8([[[primary_hue, 255, 255]]])
                #rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
                #fly['rgb_color_peak'] = rgb_color.tolist()

    fname = os.path.join(gphoto2directory, 'fly_ellipses_and_colors.pickle')
    f = open(fname, 'w')
    pickle.dump(fly_ellipses_and_colors, f)
    f.close()


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--path', type=str, default='none', help="option: path that points to directory of images")
    (options, args) = parser.parse_args()

    find_flies_in_images(options.path)
    calculate_hue_for_flies(options.path)