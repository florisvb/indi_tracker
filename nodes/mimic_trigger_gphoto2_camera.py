#!/usr/bin/env python
'''
'''
from optparse import OptionParser
import roslib
import rospy
import os
import time
import shutil
import numpy as np

from multi_tracker_analysis import read_hdf5_file_to_pandas as mta_read
from std_msgs.msg import Float32MultiArray, String

'''
sudo apt-get install libgphoto2-dev
sudo pip install -v gphoto2 (takes a while, be patient)
'''
            
# The main tracking class, a ROS node
class GPhotoCamera:
    def __init__(self, nodenum, image_source, rate=0.2, randomize_order=True):
        # where to save photos
        home_directory = os.path.expanduser( rospy.get_param('/multi_tracker/' + nodenum + '/data_directory') )
        experiment_basename = rospy.get_param('/multi_tracker/' + nodenum + '/experiment_basename', 'none')
        if experiment_basename == 'none':
            experiment_basename = time.strftime("%Y%m%d_%H%M%S_N" + nodenum, time.localtime())
        directory_name = experiment_basename + '_gphoto2'
        self.destination = os.path.join(home_directory, directory_name)
        self.image_source = image_source
        self.rate = rate
        if os.path.exists(self.destination):
            pass
        else:
            os.mkdir(self.destination)

        self.images = mta_read.get_filenames(self.image_source, '.jpg')
        if randomize_order:
            np.random.shuffle(self.images)
        self.current_image_index = -1

        # initialize the node
        rospy.init_node('gphoto2_' + nodenum)
        self.nodename = rospy.get_name().rstrip('/')
        self.nodenum = nodenum

        self.pubNewImage = rospy.Publisher('/multi_tracker/' + str(self.nodenum) + '/gphoto2_images', String)

    def gphoto_callback(self):

        self.current_image_index += 1
        try:
            image = self.images[self.current_image_index]
        except:
            rospy.signal_shutdown('All files read')
            return

        print('Captured image')
        t = rospy.Time.now()
        time_base = time.strftime("%Y%m%d_%H%M%S_N" + self.nodenum, time.localtime())
        print time_base
        name = time_base + '_' + str(t.secs) + '_' + str(t.nsecs) + '.jpg'
        target = os.path.join(self.destination, name)

        shutil.copyfile(image, target)

        self.pubNewImage.publish(String(target))
        
    def Main(self):
        rate = rospy.Rate(self.rate) # 10hz
        while not rospy.is_shutdown():
            self.gphoto_callback()
            rate.sleep()

#####################################################################################################
    
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--nodenum", type="str", dest="nodenum", default='1',
                        help="node number, for example, if running multiple tracker instances on one computer")
    parser.add_option("--image_source", type="str", dest="image_source", default='image_source',
                        help="directory where images can be found")
    parser.add_option("--rate", type="float", dest="rate", default=0.6,
                        help="rate at which to copy files")

    (options, args) = parser.parse_args()
    
    gphotocamera = GPhotoCamera(options.nodenum, options.image_source, rate=options.rate)
    gphotocamera.Main()
