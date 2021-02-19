#train modified baseline on OIDv6, while the model is already trained on CIFAR-100

import tensorflow as tf
import keras
import os
import sys
import time
import numpy as np
sys.path.insert(1, '..')
import transfer_weights

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import dense_activation, learning_rate, momentum
from class_weights_OIDv6 import class_weights

config=tf.compat.v1.ConfigProto(allow_soft_placement = True, log_device_placement = True)
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config = config)

n_classes = 64
img_dim = (64, 64, 3)

#DIRECTORIES OF DATASET
train_dir='./OID_dataset/train'
valid_dir='./OID_dataset/validation'
test_dir ='./OID_dataset/test'

def get_image_generators(batch: int):
    #data augmentation
    traingen = ImageDataGenerator(horizontal_flip=True,
                                  rotation_range=30,
                                  width_shift_range=4,
                                  height_shift_range=4,
                                  zoom_range=0.3,
                                  channel_shift_range=0.1,
                                  rescale=1./255,
                                  fill_mode='nearest')
    train_gen = traingen.flow_from_directory(train_dir,
                                             target_size=(img_dim[0], img_dim[1]),
                                             batch_size=batch,
                                             color_mode='rgb',
                                             class_mode='categorical',
                                             shuffle=True)

    datagen = ImageDataGenerator(rescale=1./255)
    valid_gen = datagen.flow_from_directory(valid_dir,
                                            target_size=(img_dim[0], img_dim[1]),
                                            batch_size=batch,
                                            color_mode='rgb',
                                            class_mode='categorical')
    test_gen = datagen.flow_from_directory (test_dir,
                                            target_size=(img_dim[0], img_dim[1]),
                                            batch_size=batch,
                                            color_mode='rgb',
                                            class_mode='categorical')
    return train_gen, valid_gen, test_gen

def get_model(old_model: str) -> tf.keras.Sequential:
    model = transfer_weights.transfer(load_model(old_model), img_dim)
    model.add(Dense(n_classes, activation=dense_activation))

    #optimizer -> stochastic gradient descent
    opt = SGD(learning_rate=learning_rate, momentum=momentum, name='SGD')

    #loss -> categorical crossentropy
    model.compile(optimizer=opt,
                  loss=CategoricalCrossentropy(),
                  metrics=[keras.metrics.CategoricalAccuracy(name='acc_'),
                           keras.metrics.TopKCategoricalAccuracy(k=5, name='top5-acc'),
                           keras.metrics.Precision(name='precision'),
                           keras.metrics.Recall(name='recall')])
    model.summary()
    return model

def start_training(old_model: str, batch_size: int, epoch_amount: int) -> None:
    model = get_model(old_model)
    train_datagen, valid_datagen, test_datagen = get_image_generators(batch_size)

    start = time.time()
    #fit data to model
    history = model.fit(train_datagen,
                        batch_size=batch_size,
                        epochs=epoch_amount,
                        validation_data=valid_datagen,
                        class_weight=class_weights,
                        use_multiprocessing=True,
                        workers=4)
    end = time.time()

    #evaluate with test-data
    eval = model.evaluate(test_datagen,
                          verbose=1)

    #save model, history of training and other data
    np.save('history_CIFAR-OIDv6.npy', history.history)
    model.save('baseline_CIFAR-OIDv6.h5')

    f = open('results_CIFAR-OIDv6.txt', 'w')
    f.write('acc, top5-acc, precision, recall\n')
    for i in eval:
        f.write("%s "% i)
    f.write('\nTime:' + str(round(end-start, 2)) + 'seconds')
    f.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('insufficien arguments: [HDF5-file] [batch] [epochs]')
    else:
        start_training(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
