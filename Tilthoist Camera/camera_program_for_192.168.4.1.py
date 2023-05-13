NO_PLC = True
import pdb
from os import write
import numpy as np
import time
import cv2
from numba import njit, prange
import threading

import pyzed.sl as sl

if NO_PLC:
    class LogixDriver(object):
        def __init__(self, ip_address):
            pass
        def __enter__(self):
            pass
        def __exit__(self, type, value, traceback):
            pass

else:
    from pycomm3 import LogixDriver

PLC_IP_ADDRESS = '192.168.4.16'
DISPLAY_HEIGHT = 720
DISPLAY_WIDTH = 1280
NUM_POINTS = 64

USE_DEPTH = True
show = True
use_fake = False

heartbeat = False

font = cv2.FONT_HERSHEY_SIMPLEX

@njit
def get_brightness_at(image, screen_coord):
    row = screen_coord[0]
    col = screen_coord[1]
    pixel_bgr = image[row][col]
    highest = 0
    for value_num in range(3):
        value = pixel_bgr[value_num]
        if value > highest:
            highest = value
    return highest


def setup_camera():
    if USE_DEPTH:
        #camera_resolution=sl.RESOLUTION.HD720,
        #camera_resolution=sl.RESOLUTION.VGA,
        init = sl.InitParameters(
            camera_resolution=sl.RESOLUTION.HD720,
            depth_mode=sl.DEPTH_MODE.ULTRA,
            coordinate_units=sl.UNIT.MILLIMETER,
            coordinate_system=sl.COORDINATE_SYSTEM.IMAGE)
        #init.camera_fps = 30
    else:
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.VGA
        init.camera_fps = 30

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, -1)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, -1)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.HUE, -1)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, -1)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, -1)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, -1)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, -1)
    return zed


@njit
def get_depth_at(depth_data, screen_coord):
    row = screen_coord[0]
    col = screen_coord[1]
    return int(depth_data[row][col])

#@njit
def get_point_distances(screen_coords_and_states, depth_data):
    point_data = np.zeros((screen_coords_and_states.shape[0],), dtype=np.int)
    for point_index, point in enumerate(screen_coords_and_states):
        point_data[point_index] = get_depth_at(depth_data, point)
    return point_data

# @njit
def get_point_brightnesses(screen_coords_and_states, image):
    point_data = np.zeros((screen_coords_and_states.shape[0],), dtype=np.int)
    for point_index, point in enumerate(screen_coords_and_states):
        point_data[point_index] = get_brightness_at(image, point)
    return point_data

@njit
def draw_blocked(coord_and_state, image):
    y = coord_and_state[0]
    x = coord_and_state[1]
    if coord_and_state[2] > 0:
        image[y-3:y+3, x-3:x+3] = np.array([0, 0, 255, 255], dtype=np.uint8)
    else:
        image[y-3:y+3, x-3:x+3] = np.array([0, 255, 0, 255], dtype=np.uint8)


@njit
def draw_states(screen_coords_and_states, image):
    for point_coord_and_state in screen_coords_and_states:
        draw_blocked(point_coord_and_state, image)


# @njit
def convert_xy_coords_to_row_column(unconverted, image_height):
    reshaped = unconverted.copy().reshape((NUM_POINTS, 3))
    output = np.zeros_like(reshaped)
    for idx, point in enumerate(reshaped):
        x = point[0]
        y = point[1]
        output[idx][0] = (image_height - y) - 1
        output[idx][1] = x
        output[idx][2] = point[2]
    return output


def write_point_data_to_plc(name, plc, point_data, heartbeat):
    try:
        plc.write(
            ("Camera1ValuesToPLC{" + str(NUM_POINTS) + "}", point_data),
            ("Camera1Heartbeat", heartbeat)
        )
    except Exception as e:
        plc = LogixDriver(PLC_IP_ADDRESS)
        plc.open()


def read_config_and_states(name, plc):
    global raw_screen_coords_and_states
    try:
        raw_screen_coords_and_states = plc.read("Camera1ValuesFromPLC{" + str(NUM_POINTS * 3) + "}").value
    except Exception as e:
        plc = LogixDriver(PLC_IP_ADDRESS)
        plc.open()


def mouse_handler(event, x, y, flags, param):
    """Update the mouse position as mouse move events call this function."""
    global mouse_xy, mouse_button_down
    VIEW_SCALE = 1
    if event == cv2.EVENT_LBUTTONDOWN:
        print("mouse down")
        mouse_button_down = True
    if event == cv2.EVENT_LBUTTONUP:
        print("mouse up")
        mouse_button_down = False
    elif event == cv2.EVENT_MOUSEMOVE:
        mouse_xy = [x,y]


if __name__ == "__main__":
    heartbeat = True
    mouse_xy = [0, 0]


    if show:
        if NO_PLC:
            cv2.namedWindow("Camera View", cv2.WINDOW_KEEPRATIO)
        else:
            cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("Camera View", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback("Camera View", mouse_handler)

    if use_fake:
        filename = f"image.npy"
        with open(filename, 'rb') as file_to_read_from:
            orig_image = np.load(file_to_read_from)
            print(f'Loaded image array from {filename}')
        filename = f"depth_data.npy"
        with open(filename, 'rb') as file_to_read_from:
            orig_depth_data = np.load(file_to_read_from)
            print(f'Loaded depth_data array from {filename}')
    else:
        zed = setup_camera()
        image_mat = sl.Mat()
        depth_mat = sl.Mat()

    raw_screen_coords_and_states = None
    retry_count = 0
    runtime = sl.RuntimeParameters()
    runtime_parameters = sl.RuntimeParameters()
    if USE_DEPTH:
        runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL

    try:

        with LogixDriver(PLC_IP_ADDRESS) as plc:
            while True:
                # threading.Thread(target=read_config_and_states, args=(1, plc), daemon=True).start()
                if not NO_PLC:
                    read_config_and_states(1, plc)
                if use_fake:
                    depth_data = orig_depth_data.copy()
                    image = orig_image.copy()
                else:
                    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                        zed.retrieve_image(image_mat, sl.VIEW.LEFT)
                        image = image_mat.get_data()
                        orig_image_shape = image.shape
                        if USE_DEPTH:
                            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                            depth_data = depth_mat.get_data()
                    else:
                        continue
                image_height = image.shape[0]

                if not NO_PLC:
                    if raw_screen_coords_and_states == None:
                        retry_count += 1
                        time.sleep(1)
                        if retry_count > 10:
                            print('Too many retrys')
                            break
                        continue

                heartbeat = not heartbeat

                if not NO_PLC:
                    # Reform the list of points as an array for faster manipulation
                    unconverted = np.array(raw_screen_coords_and_states, dtype=np.int)
                    screen_coords_and_states = convert_xy_coords_to_row_column(unconverted, image_height)

                    if USE_DEPTH:
                        # Use the depth data from the camera to create distances to send to the PLC
                        point_data = get_point_distances(screen_coords_and_states, depth_data).tolist()
                    else:
                        point_data = get_point_brightnesses(screen_coords_and_states, image).tolist()

                def fix_mouse_xy(image_shape):
                    img_y = image_shape[0]
                    img_x = image_shape[1]
                    return [int((mouse_xy[0] / DISPLAY_WIDTH) * img_x), img_y - int((mouse_xy[1] / DISPLAY_HEIGHT) * img_y)]

                # threading.Thread(target=write_point_data_to_plc, args=(1, plc, point_data, heartbeat), daemon=True).start()
                if not NO_PLC:
                    write_point_data_to_plc(1, plc, point_data, heartbeat)
                #if image.shape[0] != DISPLAY_HEIGHT or image.shape[1] != DISPLAY_WIDTH:

                if not NO_PLC:
                    draw_states(screen_coords_and_states, image)

                adjusted_mouse_xy = fix_mouse_xy(orig_image_shape)
                image = cv2.resize(image, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                image = cv2.putText(
                    image,
                    f"Mouse={adjusted_mouse_xy} {[orig_image_shape[1], orig_image_shape[0]]}",
                    (10, 460),
                    font,
                    1,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA)
                if show:
                    cv2.imshow("Camera View", image)
                    #cv2.imshow("Camera View", image)
                    key = cv2.waitKey(1)
                    if key == 27:
                        break
                    if key == 115: #S Key
                        if use_fake:
                            print("Already saved")
                        else:
                            filename = f"image.npy"
                            with open(filename, 'wb') as file_to_write_to:
                                np.save(file_to_write_to, image)
                                print(f'Saved image array to {filename}')
                            filename = f"depth_data.npy"
                            with open(filename, 'wb') as file_to_write_to:
                                np.save(file_to_write_to, depth_data)
                                print(f'Saved depth_data array to {filename}')
    finally:
        if not use_fake:
            zed.close()
            time.sleep(1)
