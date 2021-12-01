import sys
import numpy as np
import configparser
import time
import pickle

sys.path.append("../../coppelia_sim")

from API_coppeliasim import CoppeliaSim
from baseline import Baseline
from DSN import DSN
from random import randint

CONFIG_CS = '../../config/coppeliasim.ini'  #configuration of CoppeliaSim scene
CONFIG_COMMAND = '../../config/commands.ini' #configuration of command velocities
CONFIG_DSN = '../../config/DSN.ini'         #configuration of DSN, with transformations

NONE = 3 #replace none with number, for plot-purposes

def init_CoppeliaSim() -> CoppeliaSim:
    config = configparser.ConfigParser()
    config.read(CONFIG_CS)

    #address and port for remote API
    address = config['COM']['address']
    port = int(config['COM']['port'])

    CS = CoppeliaSim(address, port)

    CS.check_startup_sim() #start simulation

    #get handles
    motors = [config['HANDLES']['leftmotor'], config['HANDLES']['rightmotor']]
    sensors = [config['HANDLES']['camera'], config['HANDLES']['ultrasonic']]
    CS.get_handles(motors, sensors)
    return CS

#create DSN model, initialise it with configuration file
def init_DSN(CNN_model: str) -> DSN:
    config_DSN = configparser.ConfigParser()
    config_DSN.read(CONFIG_DSN)

    #get transformations
    transformations = []
    transformations.append([int(config_DSN['LEFT']['shift']), int(config_DSN['LEFT']['zoom'])])
    transformations.append([int(config_DSN['RIGHT']['shift']), int(config_DSN['RIGHT']['zoom'])])
    transformations.append([int(config_DSN['STRAIGHT']['shift']), int(config_DSN['STRAIGHT']['zoom'])])

    variant = int(config_DSN['GENERAL']['variant'])
    tau = float(config_DSN['GENERAL']['tau'])

    config_CS = configparser.ConfigParser()
    config_CS.read(CONFIG_CS)
    res = (int(config_CS['IMAGE']['resolution_actual']), int(config_CS['IMAGE']['resolution_actual']))

    return DSN(CNN_model, variant, transformations, tau, res)

#get motor velocities per command from configuration file
def get_commands() -> list:
    C = []
    config = configparser.ConfigParser()
    config.read(CONFIG_COMMAND)

    for section in config.sections():
        C.append([int(config[section]['leftmotor']), int(config[section]['rightmotor'])])
    return C

#execute the obstacle avoidance
#mode = 0 -> baseline
#mode = 1 -> DSN
def main_loop(CNN_model: str, mode: int, duration_target: int, output: str) -> None:
    CS = init_CoppeliaSim()
    if mode: #DSN
        model = init_DSN(CNN_model)
    else: #baseline
        model = Baseline(CNN_model)

    COMMANDS = get_commands()

    ULTRASONIC_DATA = [] #stores tuples of ultrasonic data

    #start obstacle avoidance in scene
    try:
        start = time.time()
        duration = 0
        while duration < duration_target:
            #process data from ultrasonic sensor
            current = time.time()
            duration = round(current-start, 2)
            ultrasonic_meas = CS.get_ultrasonic()

            if ultrasonic_meas == None:
                ultrasonic_meas = NONE #set higher than range of ultrasonic sensor

            ULTRASONIC_DATA.append((duration, ultrasonic_meas))

            #get image from robot
            res, img_list = CS.get_image()
            if res == -1: #no image received
                continue
            img = np.array(img_list, dtype=np.uint8)
            img.resize([1, res[0], res[1], 3]) #convert to right format
            img[0] = np.flipud(img[0]) #vertically flip img

            #determine and execute command
            command = model.determine_command(img)
            if command == 0: #backwards, randomised
                if randint(0,1) == 1:
                    CS.set_velocity(COMMANDS[command][0], -1*COMMANDS[command][1])
                else:
                    CS.set_velocity(-1*COMMANDS[command][1], COMMANDS[command][1])
            else:
                CS.set_velocity(COMMANDS[command][0], COMMANDS[command][1])

    except KeyboardInterrupt: #exit script, CTRL+C
        pass

    CS.set_velocity(0,0) #set velocity of motors to zero
    #save ultrasonic data
    with open('./data/' + output, 'wb') as fp:
        pickle.dump(ULTRASONIC_DATA, fp)

    CS.stop_simulation()
    CS.exit_API('\nClosing connection with CoppeliaSim...\nSaved ultrasonic data to ' + output)

#start of script
if __name__ == "__main__":
    if len(sys.argv) != 5: #insufficient amount of arguments provided
        print('insufficient arguments: [HDF5-file] [0/1] [time in s] [output]')
        print('0 = baseline | 1 = DSN')
    else: #start main loop
        main_loop(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
