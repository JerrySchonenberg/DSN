#configure DSN based on the velocities of the commands
import configparser
import os
import time
import tensorflow as tf
import subprocess
import numpy as np
import cv2
import sys
import math
import matplotlib.pyplot as plt

sys.path.append("../coppelia_sim")

from API_coppeliasim import CoppeliaSim
from PIL import Image


PATH_EXEC = './coppeliaSim.sh' #symbolic link

COMMAND_INIT = '../config/commands.ini'
VELOCITY = [] #velocity per command, as defined in commands.ini

CS_INIT = '../config/coppeliasim.ini'
HANDLE_NAME = [] #name of the handles

CONFIG_OUT = '../config/DSN.ini'

IMAGES = [] #store all images to compute the angles and zoom from

RESOLUTION_CONFIG = -1
RESOLUTION_ACTUAL = -1

ITER = int(sys.argv[1]) #how many times should every command be handled


#initialize the commands from a configuration file
def command_init() -> None:
    config = configparser.ConfigParser()
    config.read(COMMAND_INIT)

    backwards = True #skip backwards command
    for section in config.sections():
        if backwards == False:
            VELOCITY.append([int(config[section]['leftmotor']), int(config[section]['rightmotor'])])
        else:
            backwards = False

#start the configuration scene on coppeliasim
def scene_init() -> tuple(str, str, int):
    config = configparser.ConfigParser()
    config.read(CS_INIT)

    scene = config['COM']['scene']
    address = config['COM']['address']
    port = int(config['COM']['port'])

    for i in config['HANDLES']:
        HANDLE_NAME.append(config.get('HANDLES', i))

    global RESOLUTION_CONFIG
    RESOLUTION_CONFIG = int(config['IMAGE']['resolution_config'])
    global RESOLUTION_ACTUAL
    RESOLUTION_ACTUAL = int(config['IMAGE']['resolution_actual'])

    return scene, address, port

#get image from coppeliasim robot
def retrieve_image(CS: CoppeliaSim) -> np.ndarray:
    resolution, img_list = CS.get_image()

    img = np.array(img_list, dtype=np.uint8)
    img.resize([resolution[0], resolution[1], 3]) #convert into right format
    img = np.flipud(img) #vertically flip img
    return img

#write results to configuration file (.ini)
def write_config_init(dx: list, dy: list, DSN_variant: int, tau: float) -> None:
    config_command = configparser.ConfigParser()
    config_command.read(COMMAND_INIT)

    config_DSN = configparser.ConfigParser()

    config_DSN['GENERAL'] = {'variant' : str(DSN_variant),
                             'tau' : str(tau)}

    i = 0
    backwards = True #used to skip backwards command
    for command in config_command.sections():
        if backwards == False:
            config_DSN[command] = {'shift' : str(dx[i]),
                                   'zoom' : str(dy[i])}
            i += 1
        else:
            backwards = False

    with open(CONFIG_OUT, 'w') as configfile:
        config_DSN.write(configfile)

#use AKAZE for feature point detection
def AKAZE(DSN_variant: int, tau: float) -> None:
    pixel_ratio = RESOLUTION_CONFIG / RESOLUTION_ACTUAL
    dx = [0] * len(VELOCITY) #contains amount of horizontal pixels to be shifted
    dy = [0] * len(VELOCITY) #same as dx, but for vertical pixels

    for i in range(ITER):
        for command in range(len(VELOCITY)):
            temp_dx, temp_dy = 0, 0
            list_kp1, list_kp2 = [], []

            cv_img1 = IMAGES[i*len(VELOCITY)+command]
            cv_img2 = cv2.cvtColor(IMAGES[i*len(VELOCITY)+command+1], cv2.COLOR_RGB2GRAY)

            #AKAZE feature point detection and matching
            akaze = cv2.AKAZE_create()
            img1_kp, img1_ds = akaze.detectAndCompute(cv_img1, None)
            img2_kp, img2_ds = akaze.detectAndCompute(cv_img2, None)

            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            img1_ds = np.float32(img1_ds)
            img2_ds = np.float32(img2_ds)
            matches = flann.knnMatch(img1_ds, img2_ds, 2)

            #need to draw only good matches, so create a mask
            matchesMask = [[0,0] for i in range(len(matches))]

            #atio test as per Lowe's paper
            for j,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    matchesMask[j]=[1,0]
                    list_kp1.append([img1_kp[m.queryIdx].pt[0], img1_kp[m.queryIdx].pt[1]])
                    list_kp2.append([img2_kp[m.trainIdx].pt[0], img2_kp[m.trainIdx].pt[1]])

            count = 0
            if len(list_kp1) > 0:
                for j in range(len(list_kp1)):
                    temp_dx += list_kp2[j][0] - list_kp1[j][0]
                    if list_kp1[j][1] >= RESOLUTION_CONFIG/2: #only upper half of image considered
                        temp_dy += list_kp2[j][1] - list_kp1[j][1]
                        count += 1

                temp_dx /= len(list_kp1)
                temp_dy /= count

                dx[command] += temp_dx
                dy[command] += temp_dy

    for i in range(len(VELOCITY)):
        dx[i] = (dx[i] / ITER) / pixel_ratio
        dy[i] = (dy[i] / ITER) / pixel_ratio

        if dx[i] < 0:
            dx[i] = math.ceil(dx[i])
        else:
            dx[i] = math.floor(dx[i])

        if dy[i] < 0:
            dy[i] = math.ceil(dy[i])
        else:
            dy[i] = math.floor(dy[i])
    write_config_init(dx, dy, DSN_variant, tau)

#simulate use of CNN
def dummy_cnn() -> None:
    img = np.zeros((1,64,64,3), dtype=np.int)
    model.predict(img)
    model.predict(img)

#main loop of program
def main_loop(address: str, port: int, DSN_variant: int, tau: float) -> None:
    CS = CoppeliaSim(address, port)
    CS.get_handles(HANDLE_NAME[0:2], HANDLE_NAME[2:]) #motor-handle, sensor-handle
    CS.check_startup_sim()

    print("Configuring DSN...")

    #first image is always blank
    CS.get_image()
    CS.get_image()

    #get image of starting point
    img = retrieve_image(CS)
    IMAGES.append(img)

    for i in range(ITER):
        for command in range(len(VELOCITY)):
            CS.set_velocity(VELOCITY[command][0], VELOCITY[command][1])
            dummy_cnn()
            CS.set_velocity(0, 0)

            img = retrieve_image(CS)
            IMAGES.append(img)

    CS.stop_simulation()

    AKAZE(DSN_variant, tau) #match keypoints
    CS.exit_API('Configuration completed, saved in ' + CONFIG_OUT)

#start of script
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('insufficient arguments: [iter] [DSN-variant] [tau]')
        exit()

    #get files with configuration parameters
    command_init()
    scene, address, port = scene_init()

    model = tf.keras.models.load_model('../models/weights/weights_OAH_1.h5')

    pid = os.fork()
    if pid == 0:
        with open(os.devnull, 'wb') as devnull:
            subprocess.check_call([PATH_EXEC, '-q', '-h', scene], stdout=devnull, stderr=subprocess.STDOUT)
    else:
        time.sleep(5) #wait for coppeliasim to start
        main_loop(address, port, int(sys.argv[2]), float(sys.argv[3])) #start the configuration
