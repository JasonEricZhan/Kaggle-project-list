# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



from __future__ import absolute_import, division
import sys, os, re, csv, codecs, numpy as np, pandas as pd


from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding,Dropout,Activation,GRU,Conv1D,CuDNNGRU,CuDNNLSTM
from keras.layers import SpatialDropout1D,MaxPool1D,GlobalAveragePooling1D,RepeatVector,Add
from keras.layers import Bidirectional, GlobalMaxPool1D,BatchNormalization,concatenate,TimeDistributed,Merge,Flatten
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.optimizers import Adam,SGD,Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Layer  
from keras import initializers, regularizers, constraints  
from keras import backend as K

embed_size = 200 # how big is each word vector
max_features = 180000 # how many unique words to use (i.e num rows in embedding vector)
maxlen=180




train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')



merge=pd.concat([train,test])
df=merge.reset_index(drop=True)


merge["comment_text"]=merge["comment_text"].fillna("_na_").values


import pickle  




corpus_raw=df.comment_text

    



import time

start=time.time()


from commen_preprocess import *
from glove_twitter_preprocess import *

corpus_pre1= parallelize_dataframe(corpus_raw, multiply_columns_clean)
corpus_twitter= parallelize_dataframe(corpus_pre1, multiply_columns_glove_twitter_preprocess)
pickle.dump(corpus_twitter,open("tmp_noWordNet_twitter.pkl", "wb"))
corpus_twitter=pickle.load(open("tmp_noWordNet_twitter.pkl", "rb")) 




end=time.time()

timeStep=end-start

print("spend sencond: "+str(timeStep))




df["comment_text"]=corpus_twitter



train_cl=df[:train.shape[0]]
test_cl=df[train.shape[0]:]

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_tr = train_cl[list_classes].values      
list_sentences_train=train_cl.comment_text
list_sentences_test=test_cl.comment_text


print("....At....Tokenizer")


puncuate=r'([\.\!\?\:\,])'

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=max_features,oov_token=puncuate)
tokenizer.fit_on_texts(list(list_sentences_train)+list(list_sentences_test))

        
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)                    
                    

 

    
    
totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
print("mean length:"+ str(np.mean(totalNumWords )))
print("max length:"+ str(max(totalNumWords) ) )
print("std length:"+ str(np.std(totalNumWords )))




print(" maxlen is:"+str(maxlen))

print("number of different word:"+ str(len(tokenizer.word_index.items())))

if len(tokenizer.word_index.items()) < max_features:     
       max_features=len(tokenizer.word_index.items())
      

from keras.preprocessing import sequence
print('Pad sequences (samples x time)')



X_tr = pad_sequences(list_tokenized_train, maxlen=maxlen,padding='post')
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen,padding='post')



print('x_train shape:', X_tr.shape)
print('x_test shape:', X_te.shape)


import os, re, csv, math, codecs
print('loading word embeddings...')
embeddings_index = {}
f = codecs.open('glove.twitter.27B.200d.txt', encoding='utf-8')
from tqdm import tqdm
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))

embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in tokenizer.word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

print("complete preprocess")



import sys
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))
from keras import initializers
from keras.engine import InputSpec, Layer
from keras import backend as K

"""
From https://arxiv.org/pdf/1708.00524.pdf, 
Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm
"""
class AttentionWeightedAverage(Layer):
    """
    #Computes a weighted average of the different channels across timesteps.
    #Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

import tensorflow as tf

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#set up keras session


def lstm_attention():
 
    main_input=Input(shape=(maxlen,),name='main_input')
 
    embedded_sequences= Embedding(max_features, embed_size,weights=[embedding_matrix],trainable=False)(main_input)
   
    
    hidden_dim=100
    
    x=SpatialDropout1D(0.21)(embedded_sequences)                    #0.1
    x_lstm_1 = Bidirectional(CuDNNLSTM(hidden_dim,recurrent_regularizer=regularizers.l2(1e-5),return_sequences=True))(x)
    x_lstm_2 = Bidirectional(CuDNNLSTM(hidden_dim,recurrent_regularizer=regularizers.l2(1e-5),return_sequences=True))(x_lstm_1)
   regularizer=regularizers.l2(1e-8),return_sequences=True))(x_gru_1)
    x_com = concatenate([x_lstm_1,x_lstm_2])
    x_att_1 = AttentionWeightedAverage()(x_com)
    x_att_1= Dropout(0.225)(x_att_1)
    x= Dense(6, activation="sigmoid",kernel_regularizer=regularizers.l2(1e-8))(x_att_1)
   
    
    
    
    model = Model(inputs=main_input, outputs=x)
    nadam=Nadam(lr=0.00125, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.0035)
    model.compile(loss='binary_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy',f1_score,auc])
    print(model.summary())
    return model


batch_size = 640

#total average roc_auc: 0.9875268030202132
