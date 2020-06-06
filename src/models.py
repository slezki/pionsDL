#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras as K


def mlp(input_shape):
    
    input_layer = K.Input(shape=input_shape, name='input')
    bnorm = K.layers.BatchNormalization()(input_layer)
    dense = K.layers.Dense(64, activation='sigmoid', name='dense1', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(bnorm)
    dense = K.layers.Dense(128, activation='sigmoid', name='dense2', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(dense)
    dense = K.layers.Dense(64, activation='sigmoid', name='dense3', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(dense)

    output_layer = K.layers.Dense(1, activation='sigmoid', name='output', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(dense)

    model = K.Model(inputs=input_layer, outputs=output_layer, name='MLP')
    model.compile(optimizer=K.optimizers.Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])    
  
    return model