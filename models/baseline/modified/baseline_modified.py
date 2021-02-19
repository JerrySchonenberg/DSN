#create new instance of modified baseline, configuration of baseline defined in config.py

import tensorflow as tf
import keras

from tensorflow.keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy

from config import * #configuration of baseline (e.g. hyperparameters)

#create new instance of modified baseline
def create_model(img_dim: list, n_classes: int) -> tf.keras.Sequential:
    model = tf.keras.Sequential()
    model.add(ZeroPadding2D(padding_amount, input_shape=(img_dim[0], img_dim[1], img_dim[2])))
    model.add(Conv2D(conv1_kernels, conv_size, kernel_regularizer=l2(l=weight_decay), bias_regularizer=l2(l=weight_decay), strides=conv_stride, padding='valid', activation=conv_activation))
    model.add(MaxPooling2D(pool_size=pooling_size, strides=pooling_stride, padding='valid'))
    model.add(BatchNormalization(momentum=momentum_bn))

    model.add(ZeroPadding2D(padding_amount))
    model.add(Conv2D(conv2_kernels, conv_size, kernel_regularizer=l2(l=weight_decay), bias_regularizer=l2(l=weight_decay), strides=conv_stride, padding='valid', activation=conv_activation))
    model.add(AveragePooling2D(pool_size=pooling_size, strides=pooling_stride, padding='valid'))
    model.add(BatchNormalization(momentum=momentum_bn))

    model.add(ZeroPadding2D(padding_amount))
    model.add(Conv2D(conv3_kernels, conv_size, kernel_regularizer=l2(l=weight_decay), bias_regularizer=l2(l=weight_decay), strides=conv_stride, padding='valid', activation=conv_activation))
    model.add(BatchNormalization(momentum=momentum_bn))

    model.add(ZeroPadding2D(padding_amount))
    model.add(Conv2D(conv4_kernels, conv_size, kernel_regularizer=l2(l=weight_decay), bias_regularizer=l2(l=weight_decay), strides=conv_stride, padding='valid', activation=conv_activation))
    model.add(AveragePooling2D(pool_size=pooling_size, strides=pooling_stride, padding='valid'))

    model.add(Flatten())
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
