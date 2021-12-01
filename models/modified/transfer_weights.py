#transfer weights from old model to new model due to difference in input shape and output classes
#dense layer needs to be added when training the model
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, BatchNormalization, SpatialDropout2D, Dropout
from tensorflow.keras.regularizers import l2

from config import * #configuration of baseline (e.g. hyperparameters)

#transfer weights from one model to a new model, this is to remove the final fully connected layer
def transfer (old_model: tf.keras.Sequential, input_shape: int) -> tf.keras.Sequential:
    #build the new model
    model = tf.keras.Sequential()
    model.add(ZeroPadding2D(padding_amount, input_shape=input_shape))
    model.add(Conv2D(conv1_kernels, conv_size, kernel_regularizer=l2(l=weight_decay), bias_regularizer=l2(l=weight_decay), strides=conv_stride, padding='valid', activation=conv_activation))
    #model.add(SpatialDropout2D(rate=dropout_conv))
    model.add(MaxPooling2D(pool_size=pooling_size, strides=pooling_stride, padding='valid'))
    model.add(BatchNormalization(momentum=momentum_bn))

    model.add(ZeroPadding2D(padding_amount))
    model.add(Conv2D(conv2_kernels, conv_size, kernel_regularizer=l2(l=weight_decay), bias_regularizer=l2(l=weight_decay), strides=conv_stride, padding='valid', activation=conv_activation))
    #model.add(SpatialDropout2D(rate=dropout_conv))
    model.add(AveragePooling2D(pool_size=pooling_size, strides=pooling_stride, padding='valid'))
    model.add(BatchNormalization(momentum=momentum_bn))

    model.add(ZeroPadding2D(padding_amount))
    model.add(Conv2D(conv3_kernels, conv_size, kernel_regularizer=l2(l=weight_decay), bias_regularizer=l2(l=weight_decay), strides=conv_stride, padding='valid', activation=conv_activation))
    #model.add(SpatialDropout2D(rate=dropout_conv))
    model.add(BatchNormalization(momentum=momentum_bn))

    model.add(ZeroPadding2D(padding_amount))
    model.add(Conv2D(conv4_kernels, conv_size, kernel_regularizer=l2(l=weight_decay), bias_regularizer=l2(l=weight_decay), strides=conv_stride, padding='valid', activation=conv_activation))
    #model.add(SpatialDropout2D(rate=dropout_conv))
    model.add(AveragePooling2D(pool_size=pooling_size, strides=pooling_stride, padding='valid'))

    model.add(Flatten())

    #transfer weights
    for layer, old_layer in zip(model.layers[1:], old_model.layers[1:]):
        layer.set_weights(old_layer.get_weights())

    return model
