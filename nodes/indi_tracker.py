#!/usr/bin/env python
'''
'''
from optparse import OptionParser
import roslib
import rospy
import os
import time
import shutil
import cv2
import pickle

from multi_tracker_analysis import read_hdf5_file_to_pandas as mta_read
from std_msgs.msg import Float32MultiArray, String

import indi_tracker_analysis.find_flies_in_image_directory as find_flies
import indi_tracker_analysis.fly_align_v2 as fly_align

class IndiTracker:
    def __init__(self, nodenum):
        # initialize the node
        self.nodename = rospy.get_name().rstrip('/')
        self.nodenum = nodenum

        self.params = { 'N_images_for_median'               : 3,
                        'resize_factor'                     : 0.2,
                        'threshold'                         : 40,
                        }
        for parameter, value in self.params.items():
            try:
                p = '/multi_tracker/' + nodenum + '/gphoto2_indi_tracker/' + parameter
                self.params[parameter] = rospy.get_param(p)
            except:
                print 'Using default parameter: ', parameter, ' = ', value

        rospy.init_node('gphoto2_indi_tracker')
        self.subNewImages = rospy.Subscriber('/multi_tracker/' + nodenum + '/gphoto2_images', String, self.new_image_callback)
        self.image_directory = None
        self.image_filenames = []
        self.median_gray = None
        self.fly_data = []
        self.processed_images = []


    def calc_median_image(self):
        if len(self.image_filenames) > self.params['N_images_for_median']:
            create_median_gray = find_flies.create_median_gray_small_image_from_directory
            self.median_gray = create_median_gray(self.image_directory, 
                                                  N             = self.params['N_images_for_median'], 
                                                  resize_factor = self.params['resize_factor'])




    def new_image_callback(self, msg):
        filename = msg.data

        if self.image_directory is None:
            self.image_directory = os.path.dirname(filename)
            self.roi_directory = os.path.join(self.image_directory, 'fly_rois')
            if os.path.exists(self.roi_directory):
                pass
            else:
                os.mkdir(self.roi_directory)
        self.image_filenames.append(filename)

        if len(self.image_filenames) <= self.params['N_images_for_median']:
            return


        self.calc_median_image()
        resize_factor = self.params['resize_factor']

        for image_filename in self.image_filenames:
            if image_filename not in self.processed_images:
                image_full = cv2.imread(image_filename)
                image_full = cv2.cvtColor(image_full, cv2.COLOR_RGB2BGR)
                image_full_gray = cv2.cvtColor(image_full, cv2.COLOR_BGR2GRAY)
                image_small = cv2.resize(image_full_gray, (0,0), fx=resize_factor, fy=resize_factor) 
                ellipses = find_flies.find_fly_in_image(image_small, 
                                             self.median_gray, 
                                             threshold=self.params['threshold'], 
                                             pixels_per_mm=10, 
                                             min_fly_length_mm=1, 
                                             max_fly_ecc=5)
                
                # calculate ellipse in terms of full size image
                large_fly_ellipses = []
                for e, ellipse in enumerate(ellipses):
                    large_ellipse = ((ellipse[0][0]/resize_factor, ellipse[0][1]/resize_factor), 
                                     (ellipse[1][0]/resize_factor, ellipse[1][1]/resize_factor), 
                                     ellipse[2])
                    roi = fly_align.get_fly_roi(image_full, large_ellipse)
                    roi = fly_align.balance_luminance(roi, large_ellipse)
                    roi = fly_align.rough_align_fly_in_roi_to_vertical_position(roi, large_ellipse)

                    roi_filename = os.path.basename(image_filename).split('.')[0] + '_roi_' + str(e) + '.jpg'
                    roi_filename = os.path.join(self.roi_directory, roi_filename)
                    cv2.imwrite(roi_filename, roi)

                    fly_data = {'ellipse': large_ellipse,
                                'roi_filename': roi_filename,
                                'original_filename': image_filename}


                    self.fly_data.append(fly_data)

                self.processed_images.append(image_filename)
    
    def Main(self):
        while not rospy.is_shutdown():
            rospy.spin()
        ellipse_filename = os.path.join(self.roi_directory, 'fly_data.pickle')
        f = open(ellipse_filename, 'w')
        pickle.dump(self.fly_data, f)
        f.close()

#####################################################################################################
    
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--nodenum", type="str", dest="nodenum", default='1',
                        help="node number, for example, if running multiple tracker instances on one computer")

    (options, args) = parser.parse_args()
    
    indi_tracker = IndiTracker(options.nodenum)
    indi_tracker.Main()
