import os
import gphoto2 as gp
from optparse import OptionParser

'''
READ ME

This script is helpful for testing your gphoto cameras. Attach all cameras. Run script. 

It will print out the names and addresses of all the cameras attached, and take a photo 
with the first one in the list. 

If you have multiple cameras, use the --address option to select a specific camera based 
on the USB address that is printed out at the start of the script. 

'''

def list_availale_cameras():
    print('Cameras detected')
    print('================')
    camera_list = []
    for name, addr in gp.check_result(gp.gp_camera_autodetect()):
        camera_list.append((name, addr))
    if not camera_list:
        print('No camera detected')
    else:
       print(camera_list)
    return camera_list

#####################################################################################################
    
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--address", type="str", dest="address", default='',
                        help="USB address for the camera you want to trigger")
    (options, args) = parser.parse_args()
    
    context = gp.gp_context_new()

    camera_list = list_availale_cameras()
    if options.address == '':
        addr = camera_list[0][1]
    else:
        addr = options.address

    # make directory
    destination = os.path.expanduser( '~/gphoto_images' )
    if os.path.exists(destination):
        pass
    else:
        os.mkdir(destination)
    destination = os.path.join(destination, 'test_image.jpg')

    # initialize camera
    camera = gp.Camera()
    port_info_list = gp.PortInfoList()
    port_info_list.load()
    idx = port_info_list.lookup_path(addr)
    camera.set_port_info(port_info_list[idx])
    camera.init()
    text = camera.get_summary()
    print('')
    print('Summary')
    print('=======')
    print(str(text))

    print('')
    print('Capture')
    print('=======')
    print('Taking the photo and saving to disk')
    file_path = gp.check_result(gp.gp_camera_capture(camera, gp.GP_CAPTURE_IMAGE))
    camera_file = gp.check_result(gp.gp_camera_file_get(
            camera, file_path.folder, file_path.name,
            gp.GP_FILE_TYPE_NORMAL))
    print('Saving test image to: ' + destination)
    gp.check_result(gp.gp_file_save(camera_file, destination))
