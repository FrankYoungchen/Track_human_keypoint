import cv2
import os
import json
import math
import numpy as np
import uuid
from keras import optimizers
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Conv2D, Activation,add,UpSampling2D
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import random
import keras.backend as K
def creat_model_7_conv(w, h, s):
    input_tensor = Input(shape=(w, h, s))
    x = Conv2D(8, (3, 3), padding= 'same',activation='relu', name='Conv1')(input_tensor)
    x = Conv2D(8, (3, 3), padding='same',strides = (2,2), activation='relu', name='Conv2')(x)
    x = Conv2D(16, (3, 3), padding='same', activation='relu', name='Conv3')(x)
    x = Conv2D(16, (3, 3), padding='same',strides = (2,2), activation='relu', name='Conv4')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='Conv5')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='Conv6')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='Conv7')(x)
    output_tensor = Conv2D(1, (3, 3), padding='same', name='Conv8')(x)
    model = Model(input_tensor, output_tensor)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.RMSprop(lr=1e-3),
                  )#这个地方不知道设置什么loss
    return model
def layer(inputs, kernel, step):
    x = Conv2D(kernel, kernel_size=(1, 1), strides=(step, step), padding='same')(inputs)
    x = BatchNormalization(axis=1)(x)
    x = Activation(relu6)(x)
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation(relu6)(x)
    return x
def creat_model_mobilenetv1(w, h, s):
    input_tensor = Input(shape=(w, h, s))
    x = layer(input_tensor, 8, 1)
    x = layer(x, 8, 1)
    x = layer(x, 16, 1)
    x = layer(x, 16, 2)
    x = layer(x, 32, 1)
    x = layer(x, 32, 2)
    x = layer(x, 64, 1)
    output_tensor = Conv2D(1, (3, 3), padding='same', name='Conv8')(x)
    model = Model(input_tensor, output_tensor)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.RMSprop(lr=1e-3),
                  )
    return model
def _bottleneck(inputs, filters, kernel, t, s, r=False):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t
    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))
    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)
    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    if r:
        x = add([x, inputs])
def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    x = _bottleneck(inputs, filters, kernel, t, strides)
    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)
    return x
def creat_model_mobilenetv2(w, h, s):
    input_tensor = Input(shape=(w, h, s))
    x = _conv_block(input_tensor, 8, (3, 3), strides=(1, 1))
    x = _inverted_residual_block(x, 8, (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=4)
    #x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=2)
    #x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=1, strides=1, n=4)
    #x = _inverted_residual_block(x, 32, (3, 3), t=1, strides=1, n=2)
    #x = _inverted_residual_block(x, 32, (3, 3), t=1, strides=2, n=2)
    output_tensor = Conv2D(1, (3, 3), padding='same')(x)
    model = Model(input_tensor, output_tensor)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.RMSprop(lr=1e-3),
                  )#这个地方不知道设置什么loss
    return model
def _conv_block(inputs, filters, kernel, strides):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)
def creat_model_7_upconv(w, h, s):
    input_tensor = Input(shape=(w, h, s))
    x = Conv2D(8, (3, 3), padding= 'same',activation='relu', name='Conv1')(input_tensor)
    x = Conv2D(8, (3, 3), padding='same',strides = (2,2), activation='relu', name='Conv2')(x)
    x = Conv2D(16, (3, 3), padding='same', activation='relu', name='Conv3')(x)
    x = Conv2D(16, (3, 3), padding='same',strides = (2,2), activation='relu', name='Conv4')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='Conv5')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='Conv6')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='Conv7')(x)
    x = Conv2D(32, (3, 3), padding= 'same',activation='relu', name='Conv8')(x)
    x = UpSampling2D(size=(2, 2), data_format=None,name='UpSampling2D_1')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='Upconv_1')(x)
    x = Conv2D(16, (3, 3), padding='same', activation='relu', name='Conv9')(x)
    x = UpSampling2D(size=(2, 2), data_format=None,name='UpSampling2D_2')(x)
    x = Conv2D(16, (3, 3), padding='same', activation='relu', name='Upconv_2')(x)
    x = Conv2D(8, (3, 3), padding='same', activation='relu', name='Conv10')(x)
    output_tensor = Conv2D(1, (3, 3), padding='same', name='Conv11')(x)
    model = Model(input_tensor, output_tensor)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.RMSprop(lr=1e-3),
                  )#这个地方不知道设置什么loss
    return model