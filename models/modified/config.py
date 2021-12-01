#CONFIGURATION OF BASELINE
#excluding dimensions of input img and amount of classes

#convolutional layers
conv1_kernels=32
conv2_kernels=64
conv3_kernels=128
conv4_kernels=128
conv_size=(5, 5) #kernel size
conv_stride=1
conv_activation='relu'

#padding layers
padding_amount=2

#pooling layers
pooling_size=(3, 3)
pooling_stride=2

#dense layers
dense_activation='softmax'

#l2 regularizer
weight_decay=0.004

#stochastic gradient descent
learning_rate=0.001
momentum=0.9

#batch normalization
momentum_bn=0.4

#spatial dropout
dropout_conv=0.15
