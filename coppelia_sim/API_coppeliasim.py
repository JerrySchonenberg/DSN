#functions for remote API with CoppeliaSim
import sim

class CoppeliaSim:
    def __init__(self, address: str, port: int) -> None:
        sim.simxFinish(-1) #close all opened connections
        self.client_id = sim.simxStart(address, port, True, True, 5000, 1) #connect with CoppeliaSim

        if self.client_id == -1:
            self.exit_API('Not able to connect to CoppeliaSim API')
        print('Connected with CoppeliaSim API')

        self.HANDLE_MOTOR = []  #handle for left and right motor
        self.HANDLE_SENSOR = [] #handle for vision and ultrasonic sensor

    #check if CoppeliaSim is ready, only used for configuration of DSN
    def check_startup_sim(self) -> None:
        while True:
            if sim.simxStartSimulation(self.client_id, sim.simx_opmode_oneshot) == 0:
                break

    #end simulation run, only used for configuration of DSN
    def stop_simulation(self) -> None:
        while True:
            if sim.simxStopSimulation(self.client_id, sim.simx_opmode_oneshot) == 0:
                break

    #close connection with CoppeliaSim API
    def exit_API(self, message: str) -> None:
        print(message)
        sim.simxFinish(self.client_id)
        exit()

    #get handles of motors[left, right] and sensors[vision, ultrasonic]
    def get_handles(self, motors: list, sensors: list) -> None:
        #handle of motors
        error_l, handle_leftmotor = sim.simxGetObjectHandle(self.client_id, motors[0], sim.simx_opmode_oneshot_wait)
        error_r, handle_rightmotor = sim.simxGetObjectHandle(self.client_id, motors[1], sim.simx_opmode_oneshot_wait)
        #handle of sensors
        error_v, handle_vision = sim.simxGetObjectHandle(self.client_id, sensors[0], sim.simx_opmode_oneshot_wait)
        error_u, handle_ultrasonic = sim.simxGetObjectHandle(self.client_id, sensors[1], sim.simx_opmode_oneshot_wait)

        if error_l != 0 or error_r != 0 or error_v != 0 or error_u != 0:
            errno = str(error_l) + ',' + str(error_r) + ',' + str(error_v) + ',' + str(error_u)
            self.exit_API('Error when acquiring handle of various components, errno (l,r,vision,ultrasonic):' + str(errno))

        self.HANDLE_MOTOR = [handle_leftmotor, handle_rightmotor]
        self.HANDLE_SENSOR = [handle_vision, handle_ultrasonic]

        #initialise vision sensor
        error_v, resolution, img = sim.simxGetVisionSensorImage(self.client_id, self.HANDLE_SENSOR[0], 0, sim.simx_opmode_streaming)
        if error_v > 1:
            self.exit_API('Error with initialisation vision sensor, errno:' + str(error_v))
        #initialise ultrasonic sensor
        error_u, state, point, object_handle, normal_vector = sim.simxReadProximitySensor(self.client_id, self.HANDLE_SENSOR[1], sim.simx_opmode_streaming)
        if error_u > 1:
            self.exit_API('Error with initialisation ultrasonic sensor, errno:' + str(error_u))

    #get image from vision sensor
    def get_image(self):
        error_v, resolution, img_list = sim.simxGetVisionSensorImage(self.client_id, self.HANDLE_SENSOR[0], 0, sim.simx_opmode_buffer)

        if error_v == 0: #image received from sensor
            return resolution, img_list
        elif error_v != 1: #error
            self.exit_API('Error with vision sensor, errno:' + str(error_v))
        else: #no image received from sensor, return -1
            return -1, None

    #get measurement from ultrasonic sensor
    #range of ultrasonic sensor is 1.0m
    def get_ultrasonic(self) -> float:
        error_u, state, point, object_handle, normal_vector = sim.simxReadProximitySensor(self.client_id, self.HANDLE_SENSOR[1], sim.simx_opmode_buffer)

        if error_u == 0:
            if state == 1: #something is detected
                return point[2]
            else: #nothing detected within range
                return None
        elif error_u > 1: #error
            self.exit_API('Error with ultrasonic sensor, errno:' + str(error_u))

    #set velocity of motors
    def set_velocity(self, v_left: int, v_right: int) -> None:
        error_l = sim.simxSetJointTargetVelocity(self.client_id, self.HANDLE_MOTOR[0], v_left, sim.simx_opmode_oneshot_wait)
        error_r = sim.simxSetJointTargetVelocity(self.client_id, self.HANDLE_MOTOR[1], v_right, sim.simx_opmode_oneshot_wait)

        if error_l != 0 or error_r != 0: #error with motors
            self.exit_API('Error with motors, errno (l,r):' + str(error_l) + ',' + str(error_r))
