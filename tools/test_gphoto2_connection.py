import os
import gphoto2 as gp
from optparse import OptionParser

import indi_tracker_analysis.gphoto_utils as gphoto_utils

'''
READ ME

Install indi_tracker_analysis first.

This script is helpful for testing your gphoto cameras. Attach all cameras. Run script. 

It will print out the names and addresses of all the cameras attached, and take a photo 
with the first one in the list. 

If you have multiple cameras, use the --address option to select a specific camera based 
on the USB address that is printed out at the start of the script. 

'''

#####################################################################################################
    
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--serial", type="str", dest="serial", default='',
                        help="Serial number for the camera you want to trigger. This may not work for all cameras. Tested on Canon 5D2 and Rebel SL2")
    (options, args) = parser.parse_args()

    # check all attached cameras
    serial_to_address = gphoto_utils.list_camera_serial_numbers()
    if len(serial_to_address) == 0:
        raise ValueError('No cameras!')

    if options.serial == '':
        serial = serial_to_address.keys()[0]
    else:
        serial = options.serial


    #context = gp.gp_context_new()

    # make directory
    destination = os.path.expanduser( '~/gphoto_images' )
    if os.path.exists(destination):
        pass
    else:
        os.mkdir(destination)
    destination = os.path.join(destination, 'test_image.jpg')

    # make a camera object chosen serial number
    camera = gphoto_utils.get_camera(serial, serial_to_address)
    txt = str(camera.get_summary())
    serial = txt.split('Serial Number: ')[1].split('\n')[0]
    print('')
    print('Summary for chosen camera')
    print('=========================')
    print(str(txt))

    # Take the photo and save to disk
    print('')
    print('Capture')
    print('=======')
    print('Taking the photo and saving to disk')
    gphoto_utils.trigger_capture_and_save(camera, destination)
    print('Saved test image to: ' + destination)