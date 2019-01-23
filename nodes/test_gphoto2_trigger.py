#!/usr/bin/env python

from std_msgs.msg import Float32MultiArray
import rospy
from optparse import OptionParser



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--nodenum", type="str", dest="nodenum", default='1',
                        help="node number, for example, if running multiple tracker instances on one computer")
    (options, args) = parser.parse_args()

    rospy.init_node('test_gphoto_trigger_' + options.nodenum)

    rospy.sleep(2)
    
    pubPrefObj = rospy.Publisher('/multi_tracker/' + options.nodenum + '/prefobj', Float32MultiArray, queue_size=1)
    msg = Float32MultiArray()
    msg.data = [0, 0, 0, 0, 0]

    for i in range(2):
        pubPrefObj.publish(msg)
        rospy.sleep(1)
        print(msg)