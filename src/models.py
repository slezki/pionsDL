#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras as K


def mlp(input_shape):
    
    input_11 = K.Input(shape=(input_shape), name='input_11')
    input_12 = K.Input(shape=(input_shape), name='input_12')
    input_2 = K.Input(shape=(1,), name='input_2')

    conc = K.layers.Concatenate()([input_11, input_12])    
    bnorm = K.layers.BatchNormalization()(conc)

    dense = K.layers.Dense(64, activation='sigmoid', name='dense1', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(bnorm)
    # dense = K.layers.Dense(128, activation='sigmoid', name='dense2', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(dense)
    # dense = K.layers.Dense(64, activation='sigmoid', name='dense3', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(dense)

    conc = K.layers.Concatenate()([input_2,dense])
    bnorm = K.layers.BatchNormalization()(conc)

    dense = K.layers.Dense(16, activation='sigmoid', name='dense4', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(bnorm)
    output_layer = K.layers.Dense(1, activation='sigmoid', name='output', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(dense)

    model = K.Model(inputs=[input_11, input_12,input_2], outputs=output_layer, name='MLP')
    model.compile(optimizer=K.optimizers.Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])    
  
    return model

def mlpV2(input_shape):
    
    input_11 = K.Input(shape=(input_shape), name='input_11')
    input_12 = K.Input(shape=(input_shape), name='input_12')
    input_2 = K.Input(shape=(1,), name='input_2')

    dense1 = K.layers.Dense(32, activation='sigmoid', name='dense1', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    dense2 = K.layers.Dense(64, activation='sigmoid', name='dense2', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    # dense3 = K.layers.Dense(64, activation='sigmoid', name='dense3', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))

    dense_11 = dense1(input_11)
    dense_21 = dense2(dense_11)
    # dense_31 = dense3(dense_21)

    dense_12 = dense1(input_12)
    dense_22 = dense2(dense_12)
    # dense_32 = dense3(dense_22)

    conc = K.layers.Concatenate()([input_2,dense_21,dense_22])
    bnorm = K.layers.BatchNormalization()(conc)

    dense = K.layers.Dense(16, activation='sigmoid', name='dense4', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(conc) #(bnorm)
    output_layer = K.layers.Dense(1, activation='sigmoid', name='output', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(dense)

    model = K.Model(inputs=[input_11, input_12,input_2], outputs=output_layer, name='MLPV2')
    model.compile(optimizer=K.optimizers.Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])    
  
    return model
