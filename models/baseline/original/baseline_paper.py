#create new instance of baseline, according to the paper ("Vision based Indoor Obstacle Avoidance using a Deep Convolutional Neural Network") 
#from Khan and Parker

import tensorflow as tf
import keras

from tensorflow.keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, InputLayer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.nn import lrn # local response normalization

#HYPERPARAMETERS
weight_decay=0.004
learning_rate=0.001
momentum=0.9
#LRN specific
bias=2
depth_radius=5
alpha=0.0001
beta=0.75

#create new instance of baseline
def create_model(img_dim: list, n_classes: int) -> tf.keras.Sequential:
    model = tf.keras.Sequential()
    model.add(ZeroPadding2D(2, input_shape=(img_dim[0], img_dim[1], img_dim[2])))
    model.add(Conv2D(32, (5, 5), kernel_regularizer=l2(l=weight_decay), bias_regularizer=l2(l=weight_decay), strides=1, padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))
    model.add(InputLayer(input_tensor=lrn(model.layers[2].output, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta, name='lrn1')))

    model.add(ZeroPadding2D(2))
    model.add(Conv2D(32, (5, 5), kernel_regularizer=l2(l=weight_decay), bias_regularizer=l2(l=weight_decay), strides=1, padding='valid', activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=2, padding='valid'))
    model.add(InputLayer(input_tensor=lrn(model.layers[6].output, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta, name='lrn2')))

    model.add(ZeroPadding2D(2))
    model.add(Conv2D(64, (5, 5), kernel_regularizer=l2(l=weight_decay), bias_regularizer=l2(l=weight_decay), strides=1, padding='valid', activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=2, padding='valid'))

    model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax'))

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
