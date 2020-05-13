# essential pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2

def bn_relu_conv(input_batch, ch, k_size, name):
    '''
    realize bn + relu + conv  operation
    :param input_batch: 4D tensor
    :param ch: 1D int, the output channles
    :param k_size: 1D int, the size of convolution kernel
    '''
    x = layers.BatchNormalization()(input_batch)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(ch, k_size, padding = "same", kernel_regularizer = l2(0.001), use_bias = False, name = name)(x) # , activity_regularizer=l1(0.0002) 
    return x

def bn_relu_conv_v(input_batch, ch, k_size, name):
    '''
    realize bn + relu + conv  operation
    :param input_batch: 4D tensor
    :param ch: 1D int, the output channles
    :param k_size: 1D int, the size of convolution kernel
    '''
    x = layers.BatchNormalization()(input_batch)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(ch, k_size, padding = "valid", kernel_regularizer = l2(0.001), use_bias = False, name = name)(x) # , activity_regularizer=l1(0.0002)  
    return x

def bn_dw_conv(inputs, k_size, strs, name):
    '''
    the depthwise convolution, which also contains two operation: BN and dwconv.
    :param inputs: 3D tensor, input data
    :param k_size: 1D integer, the size of kernel
    :param strs: 1D integer, the stride for convolution
    :return: 3D tensor.
    '''
    x = layers.BatchNormalization(axis=-1, name='{}_bn'.format(name))(inputs)
    x = layers.DepthwiseConv2D(k_size, (strs, strs), padding='same', depthwise_regularizer = l2(0.001), use_bias = False, name='{}_dwconv'.format(name))(x) # depthwise_regularizer = l2(0.002),  
    return x

def bn_dw_conv_valid(inputs, k_size, strs, name):
    '''
    the depthwise convolution, which also contains two operation: BN and dwconv.
    :param inputs: 3D tensor, input data
    :param k_size: 1D integer, the size of kernel
    :param strs: 1D integer, the stride for convolution
    :return: 3D tensor.
    '''
    x = layers.BatchNormalization(axis=-1, name='{}_bn'.format(name))(inputs)
    x = layers.DepthwiseConv2D(k_size, (strs, strs), padding='valid', depthwise_regularizer = l2(0.001), use_bias = False, name='{}_dwconv'.format(name))(x) #, depthwise_regularizer = l2(0.002)  
    return x

def max_pool_S(input_batch, name):
    '''
    realize bn + relu + conv  operation
    :param input_batch: 4D tensor
    :param ch: 1D int, the output channles
    :param k_size: 1D int, the size of convolution kernel
    '''
    #max_pool = layers.AveragePooling2D(pool_size=(3, 3), strides=(1,1), padding = "same", name = name)(input_batch)
    max_pool = layers.MaxPool2D(pool_size=(3, 3), strides=(1,1), padding = "same", name = name)(input_batch)
    return max_pool

def max_pool_V(input_batch, name):
    '''
    realize bn + relu + conv  operation
    :param input_batch: 4D tensor
    :param ch: 1D int, the output channles
    :param k_size: 1D int, the size of convolution kernel
    '''
    max_pool = layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding = "valid", name = name)(input_batch)
    return max_pool

def get_inception_layer(inputs, conv11_ch, conv33_11_ch, conv33_ch, conv55_11_ch, conv55_ch, pool11_ch, name):
    '''
    A helper function to realize inception module
    :param inputs: 4D tensor
    :param conv11_ch: 1D integer, the chnnels of the branch of 1*1
    :param conv33_11_ch: 1D integer, the first chnnels of the branch of 3*3
    :param conv33_ch: 1D integer, the second chnnels of the branch of 3*3
    :param conv55_11_ch: 1D integer, the first chnnels of the branch of 5*5
    :param conv55_ch: 1D integer, the second chnnels of the branch of 5*5
    :param pool11_ch: 1D integer, the chnnels of the branch of pool
    '''

    ####################################### branch 1
    br1 = bn_relu_conv(inputs, conv11_ch, 1, name = '{}_b1'.format(name))

    ####################################### branch 2
    br2 = bn_relu_conv(inputs, conv33_11_ch, 1, name = '{}_b21'.format(name))
    br2 = bn_relu_conv(br2, conv33_ch, 3, name = '{}_b23'.format(name))

    ####################################### branch 3
    br3 = bn_relu_conv(inputs, conv55_11_ch, 1, name = '{}_b31'.format(name))
    br3 = bn_relu_conv(br3, conv55_ch, 5, name = '{}_b35'.format(name))

    ####################################### branch 4
    br4 = max_pool_S(inputs, name = '{}_pool'.format(name))
    br4 = bn_relu_conv(br4, pool11_ch, 1, name = '{}_b41'.format(name))

    ####################################### concat
    incep = layers.Concatenate(3, name = '{}_cat'.format(name))([br1, br2, br3, br4])

    return incep

def googLeNetCifar():
    '''
    The main function that defines the GoogLeNet.
    '''
    ############################################ Input
    input = layers.Input((32,32,3))
    # 32 * 32 * 3
    ############################################ two conv layer
    x = bn_relu_conv_v(input, 64, 3, name = 'C0')
    x = bn_relu_conv_v(x, 192, 3, name = 'C1')

    ########################################### two inception module
    x = get_inception_layer(x, 64, 96, 128, 16, 32, 32, name = 'incep_11')
    x = get_inception_layer(x, 128, 128, 192, 32, 96, 64, name = 'incep_12')

    ########################################## max pooling
    #x = max_pool_V(x, name = 'max_pool_1')
    ###### replace the max pooling as dwconv
    x = bn_dw_conv(x, 3, 2, name = 'dw1')
    ########################################## five inception module
    x = get_inception_layer(x, 192, 96, 208, 16, 48, 64, name = 'incep_21')
    x = get_inception_layer(x, 160, 112, 224, 24, 64, 64, name = 'incep_22')
    x = get_inception_layer(x, 128, 128, 256, 24, 64, 64, name = 'incep_23')
    x = get_inception_layer(x, 112, 144, 288, 32, 64, 64, name = 'incep_24')
    x = get_inception_layer(x, 256, 160, 320, 32, 128, 128, name = 'incep_25')
    
    ########################################## max pooling
    #x = max_pool_V(x, name = 'max_pool_2')
    ###### replace the max pooling as dwconv
    x = bn_dw_conv(x, 3, 2, name = 'dw2')
    ########################################### two inception module
    x = get_inception_layer(x, 256, 160, 320, 32, 128, 128, name = 'incep_31')
    x = get_inception_layer(x, 384, 192, 384, 48, 128, 128, name = 'incep_32')

    #x = layers.GlobalAveragePooling2D()(x)
    ############## 
    # We could not write the code by tensorflow. And the pooling and depthwise convolution are very similar in operation.
    # Hence, we apply the dpthwise convolution without the activation to realize the glpool.
    ##############
    x = bn_dw_conv_valid(x, 7, 1, name = 'glp')
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(100, activation="softmax", kernel_regularizer = l2(0.001), bias_regularizer = l2(0.001))(x) #, activity_regularizer=l1(0.0002)
    #
    model = Model(input, x)

    return model

