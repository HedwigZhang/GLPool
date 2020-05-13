import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
from  tensorflow.keras import backend as K
from tensorflow.image import ResizeMethod
#import net_utils

def channel_shuffle_layer(x):
    '''
    A helper function to realize channel shuffle
    :param x: 4D tensor
    :return: 4D tensor.
    '''
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,4,3))
    x = K.reshape(x, [-1, height, width, channels])
    return x

def bn_dwconv(inputs, k_size, st, name):
    '''
    the depthwise convolution. contains BN and depthwise convolution
    '''
    br = layers.BatchNormalization(axis=-1, name='{}_bn'.format(name))(inputs)
    br = layers.DepthwiseConv2D(k_size, strides=(st,st), padding='same', depthwise_regularizer = l2(0.002), use_bias = False, name='{}_dw'.format(name))(br)
    return br

def bn_dwconv_valid(inputs, k_size, st, name):
    '''
    the depthwise convolution. contains BN and depthwise convolution
    '''
    br = layers.BatchNormalization(axis=-1, name='{}_bn'.format(name))(inputs)
    br = layers.DepthwiseConv2D(k_size, strides=(st,st), padding='valid', depthwise_regularizer = l2(0.002), use_bias = False, name='{}_dw'.format(name))(br)
    return br

def bn_relu_conv(inputs, out_ch, k_size, st, name):
    '''
    The convolution operation. contains BN ReLU and CONV
    '''
    br = layers.BatchNormalization(axis=-1, name='{}_bn'.format(name))(inputs)
    br = layers.Activation('relu', name='{}_relu'.format(name))(br)
    br = layers.Conv2D(out_ch, k_size, strides=(st,st), padding='same', kernel_regularizer = l2(0.002), use_bias = False, name='{}_conv'.format(name))(br)
    return br

def shufflenet_v2_block_first(inputs, out_ch, name):
    '''
    '''
    sub_ch = out_ch // 2
    ############### conv branch
    conv = bn_relu_conv(inputs, sub_ch, 1, 1, name='{}_c1'.format(name))
    conv = bn_dwconv(conv, 3, 1, name='{}_dwc2'.format(name))
    conv = bn_relu_conv(conv, sub_ch, 1, 1, name='{}_c3'.format(name))
    ############### res branch
    res = bn_dwconv(inputs, 3, 1, name='{}_res2'.format(name))
    res = bn_relu_conv(inputs, sub_ch, 1, 1, name='{}_res1'.format(name))
    
    ret = layers.Concatenate(axis=-1, name='{}_concat'.format(name))([conv, res])
    #################################################################
    ret = layers.Lambda(channel_shuffle_layer, name = '{}_shuffle'.format(name))(ret)
    return ret

def shufflenet_v2_block(inputs, out_ch, name):
    '''
    '''
    sub_ch = out_ch // 2
    in_ch = inputs.get_shape().as_list()[-1]
    conv, res = layers.Lambda(channel_split, name = '{}_split'.format(name))(inputs)
    ################ res branch
    if in_ch == out_ch:
        st = 1
    else:
        st = 2
        res = bn_dwconv(res, 3, st, name='{}_res2'.format(name))
        res = bn_relu_conv(res, sub_ch, 1, 1, name='{}_res1'.format(name))
    ############### conv branch
    conv = bn_relu_conv(conv, sub_ch, 1, 1, name='{}_c1'.format(name))
    conv = bn_dwconv(conv, 3, st, name='{}_dwc2'.format(name))
    conv = bn_relu_conv(conv, sub_ch, 1, 1, name='{}_c3'.format(name))

    ret = layers.Concatenate(axis=-1, name='{}_concat'.format(name))([conv, res])
    #################################################################
    ret = layers.Lambda(channel_shuffle_layer, name = '{}_shuffle'.format(name))(ret)
    return ret


def channel_split(input_batch):
    '''
    channel split layers
    '''
    ch = input_batch.get_shape().as_list()[-1]
    sub_ch = ch // 2
    x1 = input_batch[:, :, :, 0:sub_ch]
    x2 = input_batch[:, :, :, sub_ch:ch]
    return [x1, x2]

def ShuffleNet_V2():
    '''
    This file is to designed for Cifar. The stage = {4, 8, 4}, channels = {116, 232, 464}
    '''
    input = layers.Input((32,32,3))
    # 32 * 32 * 3
    ########################################### first conv layer
    #x = layers.Conv2D(24, 3, padding = "same", kernel_regularizer = l2(0.0002), name = 'C0')(input)
    x = bn_relu_conv(input, 24, 3, 1, name='C0')
    # 32 * 32 * 24

    #################################################### stage 1
    x = shufflenet_v2_block_first(x, 116, name = 'st1_0')
    # 32 * 32 * 116
    for i in range(1, 3):
        x = shufflenet_v2_block(x, 116, name = 'st1_{}'.format(str(i)))

    ################################################### stage 2
    #x = shuffle_v2_block_second(x, 232, name = 'stage_2')
    # 16 * 16 * 232
    for i in range(8):
        x = shufflenet_v2_block(x, 232, name = 'st2_{}'.format(str(i)))
    
    ################################################### stage 3
    #x = shuffle_v2_block_second(x, 464, name = 'stage_3')
    # 8 * 8 * 464
    for i in range(4):
        x = shufflenet_v2_block(x, 464, name = 'st3_{}'.format(str(i)))
    
    ################################################### the last conv layer
    #x = layers.Conv2D(1024, kernel_size = 1, strides = (1,1), kernel_regularizer = l2(0.001), padding='same', name='1x1conv_out')(x)
    #x = bn_relu_conv(x, 1024, 1, 1, name='last')
    # 8 * 8 * 1024
    ################################################### the global average pooling layer
    #x = layers.GlobalAveragePooling2D()(x)
    ############## 
    # We could not write the code by tensorflow. And the pooling and depthwise convolution are very similar in operation.
    # Hence, we apply the dpthwise convolution without the activation to realize the glpool.
    ##############
    x = bn_dwconv_valid(x, 8, 1, name = 'glp')
    # 1 * 1 * 1024

    x = layers.Flatten()(x)
    x = layers.Dense(100, activation="softmax", kernel_regularizer = l2(0.002), bias_regularizer = l2(0.002))(x) #, activity_regularizer=l1(0.0002)
    #x = layers.Dropout(0.5)(x)

    model = Model(input, x)

    return model


