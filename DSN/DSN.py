#class for DSN, call determine_command to map image to a new command
#provide config file to specify DSN (i.e. DSN-1, DSN-2, DSN-3)

import numpy as np

from tensorflow.keras.models import load_model
from PIL import Image
from random import randint

def clamp(n: int, min_: int, max_: int) -> int:
    return max(min(max_, n), min_)

RAND = 16 #range [-RAND, RAND] in which R must be generated for shifting purposes

class DSN:
    def __init__(self, CNN_filename: str, DSN_variant: int, transformations: list, tau: float, res: int) -> None:
        self.TRANSFORMATION = transformations #transformation per command, in the form [[shift, zoom], etc..]
        self.FACTORS = [-2, 2, 0.5] #left, right, straight

        self.variant = DSN_variant #which DSN to use: DSN-1, DSN-2, etc..
        self.CNN = load_model(CNN_filename)

        self.prev_commands = [] #stores previous n commands, FIFO. n = variant
        self.prev_images = []   #stores previous n images, FIFO. n = variant

        self.tau = tau #threshold value for merging command vectors
        self.res = res #tuple of resolution (Y,X) of image, no depth

    #compute command based on current image
    #expects image in shape (1, resolution_y, resolution_x, 3)
    def determine_command(self, img: np.ndarray) -> int: 
        cv_current = self.get_command_vector(img)

        if len(self.prev_images) == self.variant and self.no_backwards():
            prediction = self.construct_prediction()
            cv_prediction = self.get_command_vector(prediction)
            cv_final = self.merge_command_vectors(cv_prediction, cv_current)
            del self.prev_images[0]
            del self.prev_commands[0]
        else: #first n images need to be skipped due to lack of previous images
            cv_final = cv_current

        command = int(np.argmax(cv_final, axis=-1)) #get entry with highest confidence
        self.prev_images.append(img)
        self.prev_commands.append(command)

        return command

    #check if the command is not executed in the last n images (n = variant)
    def no_backwards(self) -> bool:
        for i in range(self.variant):
            if self.prev_commands[i] == 0: #backwards found
                return False
        return True

    #get command vector of image from CNN
    def get_command_vector(self, img: list) -> list:
        img = np.divide(img, 255) #normalize imagedata
        return self.CNN.predict(img) #gives command_vector = [backwards, left, right, straight]

    #merges the two command vectors in the final command vectors
    def merge_command_vectors(self, cv_prediction: list, cv_current: list) -> list:
        e_prediction = 0
        e_current = 0

        for i in range(len(self.FACTORS)-1): #only left and right
            e_prediction += self.FACTORS[i] * (cv_prediction[0][i+1] < self.tau)
            e_current += self.FACTORS[i] * (cv_current[0][i+1] < self.tau)

        e_prediction *= (1 - self.FACTORS[2] * (cv_prediction[0][3] < self.tau)) #for straight
        e_current *= (1 - self.FACTORS[2] * (cv_current[0][3] < self.tau))

        if e_prediction == e_current: #no moving obstacle detected
            return cv_current
        elif e_prediction > e_current: #movement to the right detected
            cv_current[0][2] = 0.0 #set right entry to zero
            return cv_current
        else: #movement to the left detected
            cv_current[0][1] = 0.0 #set left entry to zero
            return cv_current

    #construct prediction of current image based on prev_img
    def construct_prediction(self) -> np.ndarray:
        prediction = self.prev_images[0]
        for i in range(self.variant): #apply transformations of all commands in between img_prev and img_current
            prediction[0] = self.shift_image(prediction[0], self.TRANSFORMATION[self.prev_commands[i]-1][0])
            prediction[0] = self.zoom_image(prediction[0], self.TRANSFORMATION[self.prev_commands[i]-1][1])

        return prediction

    #apply shift to image, left-shift when shift < 0, otherwise right-shift
    def shift_image(self, img: np.ndarray, shift: int) -> np.ndarray:
        result = np.zeros((self.res[0], self.res[1], 3), dtype=np.int)
        if shift > 0: #right-shift
            result[:,shift:] = img[:,:self.res[1]-shift]

            #fill in the blank pixels at the left
            for y in range(0, self.res[0]):
                for x in range(0, shift):
                    result[y, x] = self.randomise_pixel(result[y, shift+1])

        elif shift < 0: #left-shift
            shift *= -1
            result[:,:self.res[1]-shift] = img[:,shift:]

            #fill in the blank pixels ar the right
            for y in range(0, self.res[0]):
                for x in range(self.res[1]-shift, self.res[1]):
                    result[y, x] = self.randomise_pixel(result[y, self.res[1]-shift-1])

        else: #no shift
            return img
        return result

    #randomize pixel for shifting purposes
    def randomise_pixel(self, pixel: list) -> list:
        r = clamp(pixel[0] + randint(0, 2*RAND) - RAND, 0, 255) #pixel[i] + R with R in [-RAND, RAND]
        g = clamp(pixel[1] + randint(0, 2*RAND) - RAND, 0, 255)
        b = clamp(pixel[2] + randint(0, 2*RAND) - RAND, 0, 255)
        return (r,g,b)

    #apply zoom to image, only when zoom >= 0
    def zoom_image(self, img: np.ndarray, zoom: int) -> np.ndarray:
        if zoom <= 0:
            return img

        temp = Image.fromarray(img) #convert to PIL image, to use crop and resize
        temp = temp.crop((zoom, zoom, self.res[0]-zoom, self.res[1]-zoom)) #left, upper,right,lower
        result = temp.resize((self.res[0], self.res[1]))

        return np.array(result)
