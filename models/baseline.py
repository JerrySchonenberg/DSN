#class for baseline, call determine_command to map image to a new command

import numpy as np
from tensorflow.keras.models import load_model

#command_vector = [backwards, left, right, straight]
class Baseline:
    def __init__(self, CNN_filename: str) -> None:
        self.CNN = load_model(CNN_filename)

    #compute command based on current image
    #expects image in shape (1, resolution_y, resolution_x, 3)
    def determine_command(self, img: np.ndarray) -> int:
        img = np.divide(img, 255) #normalize imagedata
        return int(np.argmax(self.CNN.predict(img), axis=-1))
