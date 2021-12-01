#Assess the OAR-trained datasets on the OAH dataset, and the other way around

import tensorflow as tf
import sys

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

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

datagen = ImageDataGenerator(rescale=1./255)
test_gen = datagen.flow_from_directory(dataset,
                                       target_size=(img_dim[0], img_dim[1]),
                                       batch_size=int(sys.argv[2]),
                                       color_mode='rgb',
                                       class_mode='categorical')

eval = model.evaluate(test_gen,
                      verbose=1)

f = open('results.txt', 'w')
f.write('acc, top5-acc, precision, recall\n')
for i in eval:
    f.write("%s "% i)
f.close()
