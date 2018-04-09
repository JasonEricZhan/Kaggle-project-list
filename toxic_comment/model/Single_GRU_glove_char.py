# This Python 3 environment comes with many helpful analytics libraries installed




from __future__ import absolute_import, division
import sys, os, re, csv, codecs, numpy as np, pandas as pd


from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding,Dropout,Activation,GRU,Conv1D,CuDNNGRU,CuDNNLSTM
from keras.layers import SpatialDropout1D,MaxPool1D,GlobalAveragePooling1D,RepeatVector,Add,PReLU
from keras.layers import Bidirectional, GlobalMaxPool1D,BatchNormalization,concatenate,TimeDistributed,Merge,Flatten
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.optimizers import Adam,SGD,Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Layer  
from keras import initializers, regularizers, constraints  
from keras import backend as K


embed_size = 300 
max_features = 160000
maxlen=180


#=============

#.....preprocessing like wiki char....

#=============



class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Note: The layer has been tested with Keras 2.0.6
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number [ ] to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]





import tensorflow as tf

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#set up keras session

tf.keras.backend.set_session(sess)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


from numpy.random import seed
seed(1)




def bigru_pool_attention(attention=False):
    main_input=Input(shape=(maxlen,),name='main_input')#, name='main_input'
    Ngram_input= Input(shape=(maxlen_char,), name='aux_input')#, name='aux_input'
    embedded_sequences= Embedding(max_features, embed_size,weights=[embedding_matrix],trainable=False)(main_input)
    embedded_sequences_2= Embedding(weights_char.shape[0], 50,weights=[weights_char],trainable=True)(Ngram_input)
    
    #word level
    hidden_dim=128
    x=SpatialDropout1D(0.22)(embedded_sequences)                    #0.1
    x_gru_1 = Bidirectional(CuDNNGRU(hidden_dim,recurrent_regularizer=regularizers.l2(1e-6),return_sequences=True))(x)
    
    #char level
    hidden_dim=30
    x_2=SpatialDropout1D(0.21)(embedded_sequences_2)                    #0.1
    x_gru_2 = Bidirectional(CuDNNGRU(hidden_dim,recurrent_regularizer=regularizers.l2(1e-8),return_sequences=True))(x_2)

    
    x_ave_1=GlobalAveragePooling1D()(x_gru_1)
    x_ave_2=GlobalAveragePooling1D()(x_gru_2)
    x_ave= concatenate([x_ave_1,x_ave_2])
    x_max_1=GlobalMaxPool1D()(x_gru_1)
    x_max_2=GlobalMaxPool1D()(x_gru_2)
    x_max= concatenate([x_max_1,x_max_2])
    x_dense= concatenate([x_max,x_ave])
    
    if attention:  #did not use it at the final
       x_att_1=Attention()(x_gru_1)
       
    
    x_dense=BatchNormalization()(x_dense)
    x_dense= Dropout(0.35)(x_dense)
    x_dense = Dense(256, activation="elu")(x_dense)
    x_dense = Dropout(0.3)(x_dense)
    x_dense = Dense(128, activation="elu")(x_dense)
    x = Dropout(0.2)(x_dense)
    
    if attention:
       x=concatenate([x_att_1,x])
    
    x= Dense(6, activation="sigmoid",kernel_regularizer=regularizers.l2(1e-8))(x)
 
    
    
    
    model = Model(inputs=main_input, outputs=x)
    nadam=Nadam(lr=0.00225, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.00325)
    model.compile(loss='binary_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy',f1_score,auc])
    print(model.summary())
    return model


batch_size = 1600 #faster for char level embedding model

#total average roc_auc: 0.988312005324
