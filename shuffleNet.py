import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
from  tensorflow.keras import backend as K
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


def group_conv(inputs, in_ch, out_ch, groups, name):
    '''
    Group convolution operation
    :param inputs: 3D tensor, input
    :param in_ch: 1D integer
    :param out_ch: 1D integer
    :param groups: 1D integer, generally as 1,2,3,4,8
    '''
    sub_in_ch = in_ch // groups
    group_list = []
    for i in range(groups):
        offset = i * sub_in_ch
        group = layers.Lambda(lambda z: z[:, :, :, offset: offset + sub_in_ch], name='%s/g%d_slice' % (name, i))(inputs)
        group = bn_relu_conv(group, out_ch // groups, 1, 1, name='%s/g%d_c' % (name, i))
        group_list.append(group)

    group_conv = layers.Concatenate(name = '{}_g_cat'.format(name))(group_list)
    return group_conv

def shufflenet_bottleneck(inputs, groups, bottleneck_ratio, out_ch, name):
    '''
    The shufflenet bottleneck unit
    :param inputs: 3D tensor, input
    :param groups: 1D integer, generally as 1,2,3,4,8
    :param bottleneck_ratio: 1D integer, generally as 1,2,4.
    :param out_ch: 1D integer, the number of out channels
    '''
    in_ch = inputs.get_shape().as_list()[-1]
    
    if in_ch == out_ch:
        st = 1
        res = inputs
        bottleneck_ch = out_ch // bottleneck_ratio
    else:
        st = 2
        out_ch = out_ch // 2
        bottleneck_ch = out_ch // bottleneck_ratio
        #res = layers.AveragePooling2D(pool_size=3, strides=2, padding='same', name = '{}_pool'.format(name))(inputs)
        res = bn_dwconv(inputs, 3, st, name='{}_res_dw'.format(name))
        #res = layers.Lambda(data_pad)(res)
    
    br = group_conv(inputs, in_ch, bottleneck_ch, groups, name = '{}_gconv_1'.format(name))
    br = layers.Lambda(channel_shuffle_layer, name = '{}_shuffle'.format(name))(br)
    br = bn_dwconv(br, 3, st, name='{}_dwc2'.format(name))
    br = group_conv(br, bottleneck_ch, out_ch, groups, name = '{}_gconv_2'.format(name))

    if st == 1:
        out = layers.Add(name = '{}_add'.format(name))([br, res])
    if st == 2:
        out = layers.Concatenate(name = '{}_cat'.format(name))([br, res])

    return out

def shufflenet_bottleneck_first(inputs, groups, bottleneck_ratio, out_ch, name):
    '''
    The shufflenet bottleneck unit
    :param inputs: 3D tensor, input
    :param groups: 1D integer, generally as 1,2,3,4,8
    :param bottleneck_ratio: 1D integer, generally as 1,2,4.
    :param out_ch: 1D integer, the number of out channels
    '''
    in_ch = inputs.get_shape().as_list()[-1]
    bottleneck_ch = out_ch // bottleneck_ratio

    br = group_conv(inputs, in_ch, bottleneck_ch, groups, name = '{}_gconv_1'.format(name))
    br = layers.Lambda(channel_shuffle_layer, name = '{}_shuffle'.format(name))(br)
    br = bn_dwconv(br, 3, 1, name='{}_dwc2'.format(name))
    br = group_conv(br, bottleneck_ch, out_ch, groups, name = '{}_gconv_2'.format(name))

    res = bn_relu_conv(inputs, out_ch, 1, 1, name='{}_cr'.format(name))
    out = layers.Add(name = '{}_add'.format(name))([br, res])
    return out


def ShuffleNet():
    '''
    This file is to designed for Cifar and ImageNet32. 
    With configuration {1:144, 2:200, 3:240, 4:272, 8:384} and bottleneck_ratio = {1,2,4}, hence it is set as 4 which is widely used in bottleneck structure.
    '''
    input = layers.Input((32,32,3))
    # 32 * 32 * 3
    ########################################### first conv layer
    #x = layers.Conv2D(24, 3, padding = "same", kernel_regularizer = l2(0.0002), name = 'C0')(input)
    x = bn_relu_conv(input, 24, 3, 1, name='C0')
    # 32 * 32 * 24

    #################################################### stage 1
    x = shufflenet_bottleneck_first(x, 4, 4, 272, name = 'st1_0')
    #x = shufflenet_v2_block_first(x, 116, name = 'st1_0')
    # 32 * 32 * 116
    for i in range(1, 4):
        x = shufflenet_bottleneck(x, 4, 4, 272, name = 'st1_{}'.format(str(i)))

    ################################################### stage 2
    #x = shuffle_v2_block_second(x, 232, name = 'stage_2')
    # 16 * 16 * 232
    for i in range(8):
        x = shufflenet_bottleneck(x, 4, 4, 544, name = 'st2_{}'.format(str(i)))
    
    ################################################### stage 3
    #x = shuffle_v2_block_second(x, 464, name = 'stage_3')
    # 8 * 8 * 464
    for i in range(4):
        x = shufflenet_bottleneck(x, 4, 4, 1088, name = 'st3_{}'.format(str(i)))
    
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
    x = bn_dwconv_valid(x, 8, 1, name='glp')
    # 1 * 1 * 1024
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation="softmax", kernel_regularizer = l2(0.002), bias_regularizer = l2(0.002))(x) #, activity_regularizer=l1(0.0002)
    #x = layers.Dropout(0.5)(x)

    model = Model(input, x)

    return model


