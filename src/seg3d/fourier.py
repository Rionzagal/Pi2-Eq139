import tensorflow as tf
from tenstorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import constraints, initializers, regularizers, backend
from tensorflow.python.ops import nn

import numpy as np

class FourierOperator(Layer):
    def __init__(self, filters, num_modes, use_bias=False, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, weights_type='shared',
                 trainable=True, name=None, **kwargs):
        super.__init__(self, trainable=trainable, name=name, **kwargs)
        return 