#!/usr/bin/env python
'''
'''
from optparse import OptionParser
import roslib
import rospy
import os
import time

from std_msgs.msg import Float32MultiArray, String

'''
sudo apt-get install libgphoto2-dev
sudo pip install -v gphoto2 (takes a while, be patient)
'''
import gphoto2 as gp

import gphoto_utils
            
# The main tracking class, a ROS node
class GPhotoCamera:
    def __init__(self, nodenum, topic, serial):
        # where to save photos
        home_directory = os.path.expanduser( rospy.get_param('/multi_tracker/' + nodenum + '/data_directory') )
        experiment_basename = rospy.get_param('/multi_tracker/' + nodenum + '/experiment_basename', 'none')
        if experiment_basename == 'none':
            experiment_basename = time.strftime("%Y%m%d_%H%M%S_N" + nodenum, time.localtime())
        directory_name = experiment_basename + '_gphoto2'
        self.destination = os.path.join(home_directory, directory_name)
        if os.path.exists(self.destination):
            pass
        else:
            os.mkdir(self.destination)

        # initialize the node
        rospy.init_node('gphoto2_' + nodenum)
        self.nodename = rospy.get_name().rstrip('/')
        self.nodenum = nodenum
        self.triggers = 0

        #gp.check_result(gp.use_python_logging())
        self.camera = gphoto_utils.get_camera(serial)
        self.synchronize_camera_timestamp()
        
        self.subTrackedObjects = rospy.Subscriber('/multi_tracker/' + nodenum + '/' + topic, Float32MultiArray, self.gphoto_callback)
        self.pubNewImage = rospy.Publisher('/multi_tracker/' + str(self.nodenum) + '/gphoto2_images', String)



    def synchronize_camera_timestamp(self):
        def set_datetime(config):
            OK, date_config = gp.gp_widget_get_child_by_name(config, 'datetime')
            if OK >= gp.GP_OK:
                widget_type = gp.check_result(gp.gp_widget_get_type(date_config))
                if widget_type == gp.GP_WIDGET_DATE:
                    now = int(time.time())
                    gp.check_result(gp.gp_widget_set_value(date_config, now))
                else:
                    now = time.strftime('%Y-%m-%d %H:%M:%S')
                    gp.check_result(gp.gp_widget_set_value(date_config, now))
                return True
            return False

        # get configuration tree
        config = gp.check_result(gp.gp_camera_get_config(self.camera))#, self.context))
        # find the date/time setting config item and set it
        if set_datetime(config):
            # apply the changed config
            gp.check_result(gp.gp_camera_set_config(self.camera, config))#, self.context))
        else:
            print('Could not set date & time')
        # clean up
        gp.check_result(gp.gp_camera_exit(self.camera))#, self.context))
        return 0



    def gphoto_callback(self, msg):
        if self.triggers < 500: # max number of picturs to take
            #print('Captured image')
            t = rospy.Time.now()
            time_base = time.strftime("%Y%m%d_%H%M%S_N" + self.nodenum, time.localtime())
            name = time_base + '_' + str(t.secs) + '_' + str(t.nsecs) + '.jpg'
            destination = os.path.join(self.destination, name)

            gphoto_utils.trigger_capture_and_save(self.camera, destination)

            self.pubNewImage.publish(String(target))
            self.triggers += 1
        
    def Main(self):
        while (not rospy.is_shutdown()):
            rospy.spin()

#####################################################################################################
    
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--nodenum", type="str", dest="nodenum", default='1',
                        help="node number, for example, if running multiple tracker instances on one computer")
    parser.add_option("--topic", type="str", dest="topic", default='prefobj',
                        help="topic name (will be appended to /multi_tracker/N/). Defaults to the prefobj topic.")
    parser.add_option("--serial", type="str", dest="serial", default='',
                        help="Serial number for the camera you want to trigger. This may not work for all cameras. Tested on Canon 5D2 and Rebel SL2")
    (options, args) = parser.parse_args()
    
    gphotocamera = GPhotoCamera(options.nodenum, options.topic, options.serial)
    gphotocamera.Main()
