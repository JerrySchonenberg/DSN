#train modified baseline on CIFAR100

import tensorflow as tf
import os
import sys
import time
import numpy as np
sys.path.insert(1, '..')
import baseline_modified

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

config=tf.compat.v1.ConfigProto(allow_soft_placement = True, log_device_placement = True)
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config = config)

validation_size = 0.1 #10% of training data becomes validation data
n_classes = 20
img_dim = (32, 32, 3)

def get_image_generators():
    #load CIFAR-100 dataset, with superclasses
    (train_, train_target_), (test, test_target) = cifar100.load_data(label_mode='coarse')
    train_ = train_.astype('float32') #50000 images
    test = test.astype('float32')     #10000 images

    #reserve 10% of training data for validation set
    train = train_[:int(50000*(1-validation_size))]
    train_target =  train_target_[:int(50000*(1-validation_size))]
    validation = train_[int(50000*(1-validation_size)):]
    validation_target = train_target_[int(50000*(1-validation_size)):]

    train_target = to_categorical(train_target, n_classes)
    validation_target = to_categorical(validation_target, n_classes)
    test_target = to_categorical(test_target, n_classes)

    #create image generators
    t_datagen = ImageDataGenerator(horizontal_flip=True,
                                   rotation_range=30,
                                   width_shift_range=4,
                                   height_shift_range=4,
                                   zoom_range=0.3,
                                   channel_shift_range=0.1,
                                   rescale=1./255,
                                   fill_mode='nearest')
    train_datagen = t_datagen.flow(train, train_target, shuffle=True)

    v_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = v_datagen.flow(validation, validation_target)

    te_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = te_datagen.flow(test, test_target)

    return train_datagen, valid_datagen, test_datagen

#train and evaluate model with dataset
def start_training(batch_size: int, epoch_amount: int) -> None:
    model = baseline_modified.create_model(img_dim, n_classes)
    train_datagen, valid_datagen, test_datagen = get_image_generators()

    start = time.time()
    #fit data to model
    history = model.fit(train_datagen,
                        batch_size = batch_size,
                        epochs=epoch_amount,
                        validation_data=valid_datagen,
                        use_multiprocessing=True,
                        workers=4)
    end = time.time()

    #evaluate with test-data
    eval = model.evaluate(test_datagen,
                          verbose=1)

    #save model, history of training and other data
    np.save('history_CIFAR.npy', history.history)
    model.save('baseline_CIFAR.h5')

    f = open('results_CIFAR.txt', 'w')
    f.write('acc, top5-acc, precision, recall\n')
    for i in eval:
        f.write("%s "% i)
    f.write('\nTime:' + str(round(end-start, 2)) + 'seconds')
    f.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('insufficient arguments: [batch] [epochs]')
    else:
        start_training(int(sys.argv[1]), int(sys.argv[2]))
