import os
import tensorflow as tf
import numpy as np
from math import ceil, floor
from functools import partial
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.layers import Activation, Dropout, ZeroPadding2D, Lambda
from keras.layers import Cropping2D, Conv2DTranspose
from keras.layers import Input, Conv2D, MaxPooling2D
import keras.backend as K
import random as rn

# Parameters
BATCH_AXIS = 0
ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS=3
N_CLASS = 6
N_CHANNELS = 32
KERNEL_SIZE = (3, 3)
DROP_PROB = 0.5
POOL_SIZE = (2, 2)
N_BASE_BLOCKS = 2
MC_ITERATIONS = 10
INPUT_SHAPE = [512, 512, 1]


# Model architecture
class Reflection2Padding2D(ZeroPadding2D):
    def call(self, inputs):
        def spatial_padding(x,
                        padding=((1, 1), (1, 1)),
                        data_format=None):
            pattern = [[0, 0],list(padding[0]),list(padding[1]),[0, 0]]
            return tf.pad(inputs, pattern, mode='REFLECT')
        return spatial_padding(inputs,
                                padding=self.padding,
                                data_format=self.data_format)

class MCDropout(Dropout):
    def call(self, inputs):
        noise_shape = self._get_noise_shape(inputs)
        def dropped_inputs():
            return K.dropout(inputs, self.rate, noise_shape,
                                seed=self.seed)
        return K.in_train_phase(dropped_inputs, inputs,training=True) 

def Pad(x):
    return Reflection2Padding2D(x)

def BaseBlockLayers(x, n_channels, kernel_size):
    h = x
    for _ in range(N_BASE_BLOCKS):
        h = Pad([(k-1)//2 for k in kernel_size])(h)
        h = Conv2D(n_channels, kernel_size, padding='valid',use_bias=True)(h)
        h = BatchNormalization(axis=CHANNEL_AXIS)(h)
        h = Activation('relu')(h)
    h = partial(MCDropout(DROP_PROB))(h)

    return h

def DownBlockLayers(h_prev, n_channels, kernel_size):
    h_out   = BaseBlockLayers(h_prev, n_channels, kernel_size)
    pool_out   = MaxPooling2D(pool_size=POOL_SIZE)(h_out)
    return h_out, pool_out

def UpBlockLayers(h_prev, h_down, n_channels, kernel_size):
    h_conv   = BaseBlockLayers(h_prev, n_channels, kernel_size)
    h_prev_padded = Pad([(k-1)//2 for k in kernel_size])(h_conv)
    h_upconv = Conv2DTranspose(h_conv._keras_shape[CHANNEL_AXIS],
                    kernel_size,
                    strides=POOL_SIZE,
                    padding='valid', use_bias=True)(h_prev_padded)
    h_upconv = BatchNormalization(axis=CHANNEL_AXIS)(h_upconv)
    h_upconv = Activation('relu')(h_upconv)
    h_concat = ConcatenateLayers(h_upconv, h_down)
    return h_concat

def ConcatenateLayers(x1, x2):
    dx = (x1._keras_shape[ROW_AXIS] - x2._keras_shape[ROW_AXIS]) / 2
    dy = (x1._keras_shape[COL_AXIS] - x2._keras_shape[COL_AXIS]) / 2

    crop_size = ((floor(dx), ceil(dx)), (floor(dy), ceil(dy)))

    x12 = Concatenate(axis=CHANNEL_AXIS)([Cropping2D(crop_size)(x1), x2])

    return x12

def UNet(INPUT_SHAPE, N_CLASS, N_CHANNELS=N_CHANNELS):
    ksize = KERNEL_SIZE
    #Input
    inputs = Input(INPUT_SHAPE)
    
    #Down_1
    h_conv_1, h_pool_1 = DownBlockLayers(inputs, N_CHANNELS, ksize)
    
    #Down_2
    h_conv_2, h_pool_2 = DownBlockLayers(h_pool_1, N_CHANNELS*2, ksize)
    
    #Down_3
    h_conv_3, h_pool_3 = DownBlockLayers(h_pool_2, N_CHANNELS*4, ksize)
    
    #Down_4
    h_conv_4, h_pool_4 = DownBlockLayers(h_pool_3, N_CHANNELS*8, ksize)
    
    #Down_5_UP_5 (BottleNeck)
    h_concat_5 = UpBlockLayers (h_pool_4, h_conv_4, N_CHANNELS*16, ksize)

    #Up_4
    h_concat_4 = UpBlockLayers (h_concat_5, h_conv_3, N_CHANNELS*8, ksize)

    #Up_3
    h_concat_3 = UpBlockLayers (h_concat_4, h_conv_2, N_CHANNELS*4, ksize)
    
    #Up_2
    h_concat_2 = UpBlockLayers (h_concat_3, h_conv_1, N_CHANNELS*2, ksize)

    #Up_1
    h_conv_9 = BaseBlockLayers(h_concat_2, N_CHANNELS, ksize)
    h_conv_9_padded = Pad(1)(h_conv_9)
    #Output
    outputs = Conv2D(N_CLASS, 3, padding='valid',use_bias=True)(h_conv_9_padded)

    return Model(inputs = inputs, outputs = outputs)

def BayesianPredictor(model,INPUT_SHAPE=INPUT_SHAPE, MC_ITERATIONS=MC_ITERATIONS):
    input_shape = model.layers[0].input_shape[1:]
    inputs = Input(input_shape)
    mc_samples = Lambda(lambda x: K.repeat_elements(x, MC_ITERATIONS, axis=BATCH_AXIS))(inputs)

    logits = model(mc_samples)
    probs = Activation('softmax')(logits)

    prob   = Lambda(lambda x: K.mean(x, axis=BATCH_AXIS, keepdims=True))(probs)
    label  = Lambda(lambda x: K.argmax(x, axis=CHANNEL_AXIS))(prob)

    uncert = Lambda(lambda x: K.var(x, axis=BATCH_AXIS, keepdims=True))(probs)
    uncert = Lambda(lambda x: K.mean(x, axis=CHANNEL_AXIS))(uncert)

    return Model(inputs=inputs, outputs=[label, uncert])



