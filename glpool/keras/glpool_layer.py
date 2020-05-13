from tensorflow.keras import backend as K
import numpy as np
import math
from keras.constraints import MinMaxNorm
from keras.initializers import Constant
from keras.layers.pooling import _Pooling2D, _GlobalPooling2D
#from pooling import ow_pool
import glpool

# class _OWPooling2D(_Pooling2D):
#     """OW pooling implementation
#        Ordered Weighted Average - pooling
#        Weights are learned during the training.
#        """
#     # @interfaces.legacy_pooling2d_support
#     def __init__(self, pool_size=(7, 7),
#                  strides=None, padding='valid',
#                  data_format=None,
#                  weights_initializer='ow_avg',
#                  weights_regularizer=None,
#                  weights_constraint=None,
#                  weights_op='None',
#                  sort=True, **kwargs):
#         super(_OWPooling2D, self).__init__(pool_size, strides, padding,
#                                             data_format, **kwargs)
#         self.weights_initializer=weights_initializer
#         self.weights_regularizer=weights_regularizer
#         self.weights_constraint=weights_constraint
#         self.weights_op=weights_op
#         self.sort=sort

#     def ow_weight_initializer(self, weights_shape):
#         if self.weights_initializer == 'ow_avg':
#             ini = np.ones(weights_shape) / weights_shape[-1]
#             w_initializer = Constant(value=ini)
#         else:
#             w_initializer = self.weights_initializer
#         return w_initializer

#     def _pooling_function(self, inputs, pool_size, strides,
#                           padding, data_format):
#         outputs =  ow_pool.ow_pooling(
#                 inputs,
#                 weights = self.kernel,
#                 padding = self.padding,
#                 strides = self.strides,
#                 pool_size = self.pool_size,
#                 norm=self.weights_op,
#                 sort=self.sort)
#         return outputs

class _GLPool2D(_Pooling2D):
    """
    global learning pooling implementation
    """
    def __init__(self, pool_size=(7, 7),
                 strides = 1, padding = 'valid',
                 data_format = None,
                 weights_initializer = 'avg',
                 weights_regularizer = None,
                 weights_constraint = None,
                 weights_op = 'None', **kwargs):
        super(_GLPool2D, self).__init__(pool_size, strides, padding, data_format, **kwargs)
        self.weights_initializer = weights_initializer
        self.weights_regularizer = weights_regularizer
        self.weights_constraint = weights_constraint
        self.weights_op = weights_op
        #self.sort=sort
    
    def glpool_weight_initializer(self, weights_shape):
        if self.weights_initializer == 'avg':
            ini = np.ones(weights_shape) / weights_shape[-1]
            w_initializer = Constant(value=ini)
        else:
            w_initializer = self.weights_initializer
        return w_initializer

    def _pooling_function(self, inputs, pool_size, strides, padding, data_format):
        outputs = glpool.glp_pooling(
                inputs,
                weights = self.kernel,
                padding = self.padding,
                strides = self.strides,
                pool_size = self.pool_size,
                norm = self.weights_op
                )
        return outputs

class GLPooling2D(_GLPool2D):
    """
    global learning pooling implementation
    """

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        weights_shape = [input_dim, self.pool_size[0] * self.pool_size[1]]
        weights_initializer = self.glpool_weight_initializer(weights_shape)
        self.kernel = self.add_weight(shape = weights_shape,
                                      initializer = weights_initializer,
                                      name='kernel',
                                      regularizer = self.weights_regularizer,
                                      constraint = self.weights_constraint,
                                      trainable = True)
        super(GLPooling2D, self).build(input_shape)

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# from .. import backend as K
# from ..engine.base_layer import Layer
# from ..engine.base_layer import InputSpec
# from ..utils import conv_utils
# from ..legacy import interfaces

# class _GlobalPooling2D(Layer):
#     """Abstract class for different global pooling 2D layers.
#     """

#     @interfaces.legacy_global_pooling_support
#     def __init__(self, data_format=None, **kwargs):
#         super(_GlobalPooling2D, self).__init__(**kwargs)
#         self.data_format = K.normalize_data_format(data_format)
#         self.input_spec = InputSpec(ndim=4)

#     def compute_output_shape(self, input_shape):
#         if self.data_format == 'channels_last':
#             return (input_shape[0], input_shape[3])
#         else:
#             return (input_shape[0], input_shape[1])

#     def call(self, inputs):
#         raise NotImplementedError

#     def get_config(self):
#         config = {'data_format': self.data_format}
#         base_config = super(_GlobalPooling2D, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
