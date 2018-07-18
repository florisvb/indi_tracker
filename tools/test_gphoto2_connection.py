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

def get_list_of_availale_cameras():
    print('================')
    camera_list = []
    for name, addr in gp.check_result(gp.gp_camera_autodetect()):
        camera_list.append((name, addr))
    if not camera_list:
        print('No camera detected')
    return camera_list

def list_camera_serial_numbers():
    camera_list = get_list_of_availale_cameras()
    if len(camera_list) == 0:
        return

    addresses = [camera_info[1] for camera_info in camera_list]
    camera_types = [camera_info[0] for camera_info in camera_list]

    serial_numbers = []
    serial_to_address = {}
    for i, addr in enumerate(addresses):
        camera = gp.Camera()
        port_info_list = gp.PortInfoList()
        port_info_list.load()
        idx = port_info_list.lookup_path(addr)
        camera.set_port_info(port_info_list[idx])
        camera.init()
        txt = str(camera.get_summary())
        serial = txt.split('Serial Number: ')[1].split('\n')[0]
        serial_numbers.append(serial)
        serial_to_address[serial_numbers[i]] = addresses[i]

    print('Attached Cameras:')
    print('================')
    for i in range(len(addresses)):
        print('Camera ' + str(i+1))
        print('      Serial number: ' + serial_numbers[i])
        print('      Make and Model: ' + camera_types[i])
        print('      USB Address: ' + addresses[i])

    return serial_to_address

#####################################################################################################
    
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--serial", type="str", dest="serial", default='',
                        help="Serial number for the camera you want to trigger. This may not work for all cameras. Tested on Canon 5D2 and Rebel SL2")
    (options, args) = parser.parse_args()

    # check all attached cameras
    serial_to_address = list_camera_serial_numbers()
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
    addr = serial_to_address[serial]
    camera = gp.Camera()
    port_info_list = gp.PortInfoList()
    port_info_list.load()
    idx = port_info_list.lookup_path(addr)
    camera.set_port_info(port_info_list[idx])
    camera.init()
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
    file_path = gp.check_result(gp.gp_camera_capture(camera, gp.GP_CAPTURE_IMAGE))
    camera_file = gp.check_result(gp.gp_camera_file_get(
            camera, file_path.folder, file_path.name,
            gp.GP_FILE_TYPE_NORMAL))
    print('Saving test image to: ' + destination)
    gp.check_result(gp.gp_file_save(camera_file, destination))
