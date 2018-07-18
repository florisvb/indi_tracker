from multi_tracker_analysis import read_hdf5_file_to_pandas as mta_read

import cv2
import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
import matplotlib

from scipy.stats import itemfreq

import copy
from optparse import OptionParser

import progressbar
import dask

from distutils.version import LooseVersion, StrictVersion
print 'Using open cv: ' + cv2.__version__
if StrictVersion(cv2.__version__.split('-')[0]) >= StrictVersion("3.0.0"):
    OPENCV_VERSION = 3
    print 'Open CV 3'
else:
    OPENCV_VERSION = 2
    print 'Open CV 2'

class FlyImg(object):
    def __init__(self, directory, filename, pixels_per_mm, resize_factor=0.2):
        self.directory = directory
        self.filename = os.path.basename(filename)
        self.pixels_per_mm = float(pixels_per_mm) # for the full size image!
        self.resize_factor = resize_factor

        self.img = None
        self.gray_img = None
        self.small_gray = None
        self.small_color = None

        self.fly_ellipses_small = None
        self.fly_ellipses_large = None
        self.rois_fly = None
        self.rois_isolated_fly = None
        self.rois_median = None

        self.get_image_size()

    def path(self):
        return os.path.join(self.directory, self.filename)

    def get_image_size(self):
        cmd = 'identify -ping ' + self.path()
        output = os.popen(cmd).read()
        size = output.split('JPEG')[-1].lstrip().split(' ')[0]
        height, width = size.split('x')
        self.width = int(width)
        self.height = int(height)

    def load_img(self):
        if self.img is None:
            self.img = cv2.imread(self.path())
        return self.img

    def load_gray_img(self):
        if self.gray_img is None:
            self.gray_img = cv2.imread(self.path(), cv2.CV_8UC1)
        return self.gray_img

    def clear_images(self):
        for obj in ['img', 'gray_img', 'small_gray', 'small_color']:
            self.__delattr__(obj)
            self.__setattr__(obj, None)

    def load_small_gray_image(self):
        if self.small_gray is None:
            img_gray = self.load_gray_img()
            self.small_gray = cv2.resize(img_gray, (0,0), fx=self.resize_factor, fy=self.resize_factor) 
        return self.small_gray

    def load_small_color_image(self):
        if self.small_color is None:
            img = self.load_img()
            self.small_color = cv2.resize(img, (0,0), fx=self.resize_factor, fy=self.resize_factor) 
        return self.small_color

    def pixels_to_mm(self, length, size):
        '''
        length - pixels (e.g. length of a line in pixels)
        size - small or large
        '''

        if size == 'small':
            return (length/self.resize_factor)/self.pixels_per_mm
        elif size == 'large':
            return (length)/self.pixels_per_mm

    def save_fly_ellipses(self, fly_ellipses):
        self.fly_ellipses_small = fly_ellipses
        self.fly_ellipses_large = []

        for fly_ellipse in self.fly_ellipses_small:
            large_center = (fly_ellipse[0][0]/self.resize_factor, fly_ellipse[0][1]/self.resize_factor) 
            large_size = (fly_ellipse[1][0]/self.resize_factor, fly_ellipse[1][1]/self.resize_factor) 
            large_angle = fly_ellipse[2]
            self.fly_ellipses_large.append( (large_center, large_size, large_angle) )

    def find_flies(self, median, threshold=50, min_fly_length_mm=2.5, max_fly_ecc=5):
        '''
        median - median small gray image

        saves fly positions to self.fly_ellipses_small and self.fly_ellipses_large
        '''

        image = self.load_small_gray_image()

        absdiff = cv2.absdiff(image, median)
        retval, threshed = cv2.threshold(absdiff, threshold, 255, 0)
        
        kern_d = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_d,kern_d))
        threshed = cv2.morphologyEx(threshed,cv2.MORPH_OPEN, kernel, iterations = 1)
        
        kernel = np.ones((3,3), np.uint8)
        threshed = cv2.dilate(threshed, kernel, iterations=5)
        threshed = cv2.erode(threshed, kernel, iterations=3)

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
                fly_length = self.pixels_to_mm(np.max(ellipse[1]), 'small')
                fly_width = self.pixels_to_mm(np.min(ellipse[1]), 'small')
                fly_ecc = fly_length / fly_width
                if fly_length > min_fly_length_mm and fly_ecc < max_fly_ecc:
                    fly_ellipses.append(ellipse)
        
        self.save_fly_ellipses(fly_ellipses)

    def show_fly_ellipses(self, size='small'):
        if size == 'small':
            img = copy.copy(self.load_small_color_image())
            for fly_ellipse in self.fly_ellipses_small:
                img = cv2.ellipse(img, fly_ellipse, color=(0,255,0), thickness=5)

        elif size =='large':
            img = copy.copy(self.load_color_image())
            for fly_ellipse in self.fly_ellipses_large:
                img = cv2.ellipse(img, fly_ellipse, color=(0,255,0), thickness=5)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img)

    def load_roi(self, img, fly_ellipse, width=None):
        if width is None:
            r = int(np.max(fly_ellipse[1])/1.5)
        else:
            r = width

        _l = np.max([0, fly_ellipse[0][1]-r])
        _r = np.min([self.height, fly_ellipse[0][1]+r])
        _width = _r - _l
        _b = np.max([0, fly_ellipse[0][0]-r])
        _t = np.min([self.width, fly_ellipse[0][0]+r])
        _height = _t - _b
        zoom = img[int(_l):int(_r), int(_b):int(_t), :]

        return zoom

    def load_rois_fly(self, width=None):
        self.rois_fly = []
        if self.img is None:
            self.load_img()
        for fly_ellipse in self.fly_ellipses_large:
            zoom = self.load_roi(self.img, fly_ellipse, width=width)
            self.rois_fly.append( copy.deepcopy(zoom) )

    def load_rois_median(self, median, width=None):
        self.rois_median = []
        for fly_ellipse in self.fly_ellipses_large:
            zoom = self.load_roi(median, fly_ellipse, width=width)
            self.rois_median.append( copy.deepcopy(zoom) )

    def load_rois_isolated_fly(self):
        # run load_rois_fly and load_rois_median first
        self.rois_isolated_fly = []
        for i in range(len(self.rois_fly)):
            fly_c = self.remove_bg_from_fly_roi(self.rois_fly[i], self.rois_median[i], self.fly_ellipses_large[i])
            self.rois_isolated_fly.append( fly_c )

    def show_rois(self, rois=None):
        if rois is None:
            rois = self.rois_fly

        N = len(rois)
        n = int(np.ceil(np.sqrt(N)))

        fig = plt.figure()
        gs = matplotlib.gridspec.GridSpec(n, n)

        for i, roi in enumerate(rois):
            ax = plt.subplot(gs[i])
            ax.imshow( self.convert_bgr_to_rgb(roi) )

    def convert_bgr_to_rgb(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def remove_bg_from_fly_roi(self, fly_roi, median_roi, ellipse_large, threshold=40):

        # draw ellipse mask
        actual_width = np.max(fly_roi.shape) # won't work for corners
        ellipse_large_centered = (( int(actual_width/2.), int(actual_width/2.)),
                                    ellipse_large[1], ellipse_large[2])
        mask = np.zeros_like(fly_roi)
        mask = cv2.ellipse(mask,ellipse_large_centered,[1,1,1],-1)
        inverted_mask = 1-mask

        # get most common color in background
        median_background = get_dominant_color(median_roi, 2)
        background = np.ones_like(fly_roi)*median_background

        # isolate the fly
        isolated_fly = fly_roi*mask + inverted_mask*background

        #####

        #gray_fly = cv2.cvtColor(isolated_fly, cv2.COLOR_BGR2GRAY)
        #gray_bg = cv2.cvtColor(median_roi, cv2.COLOR_BGR2GRAY)

        absdiff = cv2.absdiff(isolated_fly, median_roi)
        retval, threshed = cv2.threshold(absdiff, 10, 255, 0)

        threshed = np.mean(threshed, axis=2)

        isolated_fly[np.where(threshed==0)] = median_background

        return isolated_fly

def get_dominant_color(img, n_colors):
    arr = np.float32(img)
    pixels = arr.reshape((-1, 3))

    n_colors = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    #quantized = palette[labels.flatten()]
    #quantized = quantized.reshape(img.shape)

    dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]

    return dominant_color

def create_median_gray_small_image_from_directory(directory, N=10):
    # N is the number of equidistant files to use for making the median
    file_list = mta_read.get_filenames(directory, '.jpg')
    imgs = []
    indices = np.linspace(0,len(file_list)-1,N).astype(int)

    for i in indices:
        file = file_list[i]
        small = dask.delayed(FlyImg(directory, file, pixels_per_mm=42).load_small_gray_image())
        imgs.append(small)

    imgs = dask.compute(*imgs)

    median = np.median(imgs, axis=0)
    return median.astype(np.uint8)

def create_median_image_from_directory(directory, N=10):
    # N is the number of equidistant files to use for making the median
    file_list = mta_read.get_filenames(directory, '.jpg')
    imgs = []
    indices = np.linspace(0,len(file_list)-1,N).astype(int)

    for i in indices:
        file = file_list[i]
        img = dask.delayed(FlyImg(directory, file, pixels_per_mm=42).load_img())
        imgs.append(img)

    imgs = dask.compute(*imgs)

    median = np.median(imgs, axis=0)
    return median.astype(np.uint8)

def find_flies_and_load_rois(directory, file, pixels_per_mm, median, median_large, width=None):
    print file
    flyimg = FlyImg(directory, file, pixels_per_mm)
    flyimg.find_flies(median)
    flyimg.load_rois_fly(width=250)
    flyimg.load_rois_median(median_large, width=250)
    flyimg.load_rois_isolated_fly()
    flyimg.clear_images()

    new_directory = os.path.join(os.path.dirname(directory), 'flyimgs')
    if not os.path.exists(new_directory):
    	os.mkdir(new_directory)

    fname = os.path.join(new_directory, os.path.basename(file).split('.')[0] + '_flyimg.pickle')
    print fname
    print
    f = open(fname, 'w')
    pickle.dump(flyimg, f)
    f.close()


def extract_all_flyimgs(directory, pixels_per_mm=42, width=None):

    file_list = mta_read.get_filenames(directory, '.jpg')
    median = create_median_gray_small_image_from_directory(directory)
    median_large = create_median_image_from_directory(directory)

    delayed_results = []
    for file in file_list:
        queue = dask.delayed(find_flies_and_load_rois)(directory, file, pixels_per_mm, median, median_large, width=width) 
        delayed_results.append(queue)

    results = dask.compute(*delayed_results)
    

### Analysis ###############################################################################################

def load_all_flyimgs(flyimgs_directory):
    flyimgs = []

    file_list = mta_read.get_filenames(flyimgs_directory, 'flyimg.pickle')

    def open_flyimg(file):
        f = open(file)
        flyimg = pickle.load(f)
        f.close()
        return flyimg

    for file in file_list:
        flyimgs.append( dask.delayed(open_flyimg)(file) ) 

    flyimgs = dask.compute(*flyimgs)

    return flyimgs

def compile_flyimg_list_to_dict(flyimg_list):
    flyimg_dict = {}
    for flyimg in flyimg_list:
        flyimg_dict[flyimg.filename] = flyimg
    return flyimg_dict

def load_flyimg_dict_from_path(path):
    if 'flyimgs' not in path:
        flyimg_path = os.path.join(path, 'flyimgs')
    else:
        flyimg_path = path
    flyimgs = load_all_flyimgs(flyimg_path)
    flyimg_dict = compile_flyimg_list_to_dict(flyimgs)
    return flyimg_dict

def save_images_from_flyimgs(path):
    flyimg_dict = load_flyimg_dict_from_path(path)

    def write_images(basename, roi, bkgrd, isolated):
        filename = basename.split('.')[0] + '_' + str(i) + '.jpg'
        filename = os.path.join( os.path.join(path, 'rois'), filename)
        cv2.imwrite(filename, roi)

        filename = basename.split('.')[0] + '_' + str(i) + '.jpg'
        filename = os.path.join( os.path.join(path, 'bkgrds'), filename)
        cv2.imwrite(filename, bkgrd)

        filename = basename.split('.')[0] + '_' + str(i) + '.jpg'
        filename = os.path.join( os.path.join(path, 'isolated'), filename)
        print filename
        cv2.imwrite(filename, isolated)

        return filename

    results = []
    for flyimg in flyimg_dict.values():
        flyimg.load_rois_isolated_fly()
        for i, roi in enumerate(flyimg.rois_fly): 
            queue = dask.delayed(write_images)(flyimg.filename, roi, flyimg.rois_median[i], flyimg.rois_isolated_fly[i])
            results.append(queue)
    results = dask.compute(*results)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--path', type=str, default='none', help="option: path that points to directory of images")
    parser.add_option('--ppm', type=int, default=42, help="option: pixels per mm for images")
    parser.add_option('--width', type=int, default=-1, help="option: roi width, -1 means none (automatic)")
    (options, args) = parser.parse_args()

    if options.width == -1:
        width = None    
    else:
        width = options.width

    flyimgs = extract_all_flyimgs(options.path, pixels_per_mm=options.ppm, width=width)

    #destination = os.path.join(options.path, 'flyimgs.pickle')
    #save_flyimgs(flyimages, destination)