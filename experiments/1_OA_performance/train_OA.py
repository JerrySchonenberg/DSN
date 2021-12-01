#finetune CNN and train it with OA dataset
#0 = OAR | 1 = OAH

import tensorflow as tf
import keras
import os
import sys
import time
import numpy as np
sys.path.insert(1, '../create_model')
import transfer_weights

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import dense_activation, learning_rate, momentum

config=tf.compat.v1.ConfigProto(allow_soft_placement = True, log_device_placement = True)
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config = config)

n_classes = 4
img_dim = (64, 64, 3)

#DIRECTORIES OF DATASETS
#OAR
OAR_train_dir='./OAR_dataset/train'
OAR_valid_dir='./OAR_dataset/validation'
OAR_test_dir ='./OAR_dataset/test'
#OAH
OAH_train_dir='./OAH_dataset/train'
OAH_valid_dir='./OAH_dataset/validation'
OAH_test_dir ='./OAH_dataset/test'

#==============================================================

def get_image_generators(train_dir, valid_dir, test_dir, batch):
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(train_dir,
                                            target_size=(img_dim[0], img_dim[1]),
                                            batch_size=batch,
                                            color_mode='rgb',
                                            class_mode='categorical',
                                            shuffle=True)
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

#==============================================================

def finetune_model(old_model):
    model = transfer_weights.transfer(load_model(old_model), img_dim)
    model.trainable = False #freeze layers to fine-tune model
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

#==============================================================

def start_training(old_model, batch_size, epoch_amount, dataset):
    model = finetune_model(old_model)
    if dataset: #OAH dataset
        train_datagen, valid_datagen, test_datagen = get_image_generators(OAH_train_dir,
                                                                          OAH_valid_dir,
                                                                          OAH_test_dir,
                                                                          batch_size)
    else: #OAR dataset
        train_datagen, valid_datagen, test_datagen = get_image_generators(OAR_train_dir,
                                                                          OAR_valid_dir,
                                                                          OAR_test_dir,
                                                                          batch_size)

    start = time.time()
    #fit data to model
    history = model.fit(train_datagen,
                        batch_size=batch_size,
                        epochs=epoch_amount,
                        validation_data=valid_datagen,
                        use_multiprocessing=True,
                        workers=4)
    end = time.time()

    #evaluate with test-data
    eval = model.evaluate(test_datagen,
                          verbose=1)

    #save model, history of training and other data
    np.save('history_OA.npy', history.history)
    model.save('baseline_OA.h5')

    f = open('results_OA.txt', 'w')
    f.write('acc, top5-acc, precision, recall\n')
    for i in eval:
        f.write("%s "% i)
    f.write('\nTime:' + str(round(end-start, 2)) + 'seconds')
    f.close()

#==============================================================

if len(sys.argv) != 5:
    print('insufficien arguments: [HDF5-file] [batch] [epochs] [0/1]')
    print('0 = OAR dataset | 1 = OAH dataset')
else:
    start_training(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
