#get TPR and FPR metrics from networks

import tensorflow as tf
import sys
import keras
sys.path.insert(1, '../../baseline/create_model')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from config import *

#test images from both datasets
OAR = './OAR_dataset/test'
OAH = './OAH_dataset/test'

img_dim = (64,64,3)

#=========================================================

if len(sys.argv) != 4:
    print('insufficient arguments: [HDF5-file] [batch] [0/1]')
    print('0 = OAR dataset | 1 = OAH dataset')

if int(sys.argv[3]) == 0:
    dataset = OAR
else:
    dataset = OAH

model = load_model(sys.argv[1])

#optimizer -> stochastic gradient descent
opt = SGD(learning_rate=learning_rate, momentum=momentum, name='SGD')
#loss -> categorical crossentropy
model.compile(optimizer=opt,
              loss=CategoricalCrossentropy(),
              metrics=[keras.metrics.TruePositives(name='tp'),
                       keras.metrics.FalsePositives(name='fp'),
                       keras.metrics.TrueNegatives(name='tn'),
                       keras.metrics.FalseNegatives(name='fn')])

datagen = ImageDataGenerator(rescale=1./255)
test_gen = datagen.flow_from_directory(dataset,
                                       target_size=(img_dim[0], img_dim[1]),
                                       batch_size=int(sys.argv[2]),
                                       color_mode='rgb',
                                       class_mode='categorical')

eval = model.evaluate(test_gen,
                      verbose=1)

TPR = eval[1] / (eval[1] + eval[4])
FPR = eval[2] / (eval[2] + eval[3])

print(TPR)
print(FPR)
