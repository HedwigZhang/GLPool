###################
# The implemation of ResNet
###################

import tensorflow as tf
#from tensorflow.keras import layers
from tensorflow.keras import layers
#Input, Conv2d, AveragePooling2D, Add, BatchNormalization, Activation, GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2

def bn_relu_conv(input_batch, ch, k_size, st, name):
    '''
    realize conv + bn + relu operation
    :param input_batch: 4D tensor
    :param ch: 1D int, the channles
    :param k_size: 1D int, the size of convolution kernel
    '''
    x = layers.BatchNormalization()(input_batch)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(ch, k_size, strides=(st, st), padding = "same", kernel_regularizer = l2(0.004), use_bias = False, name = name)(x) # , activity_regularizer=l1(0.0002) , kernel_regularizer = l2(0.002), bias_regularizer = l2(0.002)
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
    x = layers.DepthwiseConv2D(kernel_size = k_size, strides=(strs, strs), padding='same', depthwise_regularizer = l2(0.004), use_bias = False, name='{}_dwconv'.format(name))(x) #  depthwise_regularizer = l2(0.002), use_bias = False,
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
    x = layers.DepthwiseConv2D(kernel_size = k_size, strides=(strs, strs), padding='valid', depthwise_regularizer = l2(0.004), use_bias = False, name='{}_dwconv'.format(name))(x) #  depthwise_regularizer = l2(0.002), use_bias = False,
    return x


def data_pad(x):
    ch = x.get_shape().as_list()[-1]
    sub_ch = ch // 2
    x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [sub_ch, sub_ch]])
    return x   

# def basic_block(inputs, out_ch, name):
#     '''
#     The basic bloc of ResNet, which has the configuration {2,2,2,2} ResNet18 and {3,4,6,3} ResNet34.
#     :param inputs: 3D tesnor, the input data
#     :param out_ch: 1D integer, the output channels
#     return 3D tensor
#     '''
#     in_ch = inputs.get_shape().as_list()[-1]
#     if in_ch == out_ch:
#         st = 1
#         res = inputs
#     else:
#         st = 2
#         res = layers.AveragePooling2D(2)(inputs)
#         res = layers.Lambda(data_pad)(res)
    
#     br = bn_relu_conv(inputs, out_ch, 3, st, name = '{}_c1'.format(name)) #### 32*32*256
#     br = bn_relu_conv(br, out_ch, 3, 1, name = '{}_c2'.format(name))
#     out = layers.Add(name='{}_add'.format(name))([br, res])  #### 32*32*256
#     return out

def bottleneck_dw_block(inputs, out_ch, name):
    '''
    The bottlneck block of the ResNet, which has the configuration of {3,4,6,3} ResNet50 and {3,4,23,3} ResNet101
    :param inputs: 3D tesnor, the input data
    :param out_ch: 1D integer, the output channels
    return 3D tensor
    '''
    ############## here the in_ch = 64, out_ch = 256. Hence the input is 32*32*64
    sub_ch = out_ch // 4
    in_ch = inputs.get_shape().as_list()[-1]
    if in_ch == out_ch:
        st = 1
        res = inputs
    else:
        st = 2
        res = bn_dw_conv(inputs, 3, st, name = '{}_dw'.format(name))
        res = layers.Lambda(data_pad)(res)
    ############ ResNet
    #res = bn_relu_conv(inputs, out_ch, 1, 1, name = '{}_res'.format(name)) #### 32*32*256
    ############## conv
    br = bn_relu_conv(inputs, sub_ch, 1, 1, name = '{}_c1'.format(name))  #### 32*32*64
    br = bn_relu_conv(br, sub_ch, 3, st, name = '{}_c2'.format(name))      #### 32*32*64
    br = bn_relu_conv(br, out_ch, 1, 1, name = '{}_c3'.format(name))      #### 32*32*256
    ############## The resdiaul
    out = layers.Add(name='{}_add'.format(name))([br, res])  #### 32*32*256
    return out

def bottleneck_block_stage1(inputs, out_ch, name):
    '''
    The bottlneck block of the ResNet, which has the configuration of {3,4,6,3} ResNet50 and {3,4,23,3} ResNet101
    :param inputs: 3D tesnor, the input data
    :param out_ch: 1D integer, the output channels
    return 3D tensor
    '''
    sub_ch = out_ch // 4
    res = bn_relu_conv(inputs, out_ch, 1, 1, name = '{}_res'.format(name)) #### 32*32*256
    ############## conv
    br = bn_relu_conv(inputs, sub_ch, 1, 1, name = '{}_c1'.format(name))  #### 32*32*64
    br = bn_relu_conv(br, sub_ch, 3, 1, name = '{}_c2'.format(name))      #### 32*32*64
    br = bn_relu_conv(br, out_ch, 1, 1, name = '{}_c3'.format(name))      #### 32*32*256
    ############## The resdiaul
    out = layers.Add(name='{}_add'.format(name))([br, res])  #### 32*32*256
    return out

def resnet():
    '''
    realize the Residual inception Net
    18: {2,2,2,2} basic block
    34: {3,4,6,3} basic block
    50: {3,4,6,3} bottleneck block
    101: {3,4,23,3} bottleneck block
    '''
    # Input
    input = layers.Input((32,32,3))   # 32 * 32 * 3
    ## first conv layer
    x = bn_relu_conv(input, 64, 3, 1, name = 'C0')  # 32 * 32 * 64
    ############# stage1
    x = bottleneck_block_stage1(x, 256, name = 'stage1') # 32 * 32 * 256
    for j in range(2):
        x = bottleneck_dw_block(x, 256, name = 'st1_'+ str(j))

    ############# stage2
    x = bottleneck_dw_block(x, 512, name = 'stage2') # 16 * 16 * 512
    for j in range(3):
        x = bottleneck_dw_block(x, 512, name = 'st2_'+ str(j))

    ############# stage3
    x = bottleneck_dw_block(x, 1024, name = 'stage3') # 8 * 8 * 1024
    for j in range(5):
        x = bottleneck_dw_block(x, 1024, name = 'st3_'+ str(j))

    ############# stage4
    x = bottleneck_dw_block(x, 2048, name = 'stage4') # 4 * 4 * 2048
    for j in range(2):
        x = bottleneck_dw_block(x, 2048, name = 'st4_'+ str(j))
    
    ############## 
    # We could not write the code by tensorflow. And the pooling and depthwise convolution are very similar in operation.
    # Hence, we apply the dpthwise convolution without the activation to realize the glpool.
    ##############
    x = bn_dw_conv_valid(x, 4, 4, name = 'glp')
    # 1 * 1 * 128
    x = layers.Flatten()(x)
    #x = layers.Dropout(0.5)(x)
    x = layers.Dense(100, kernel_regularizer = l2(0.004), bias_regularizer = l2(0.004), activation="softmax")(x) #, activity_regularizer=l1(0.0002) kernel_regularizer = l2(0.002), bias_regularizer = l2(0.002)
    model = Model(input, x)
    return model