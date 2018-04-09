# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



from __future__ import absolute_import, division
import sys, os, re, csv, codecs, numpy as np, pandas as pd


from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding,Dropout,Activation,GRU,Conv1D,CuDNNGRU,CuDNNLSTM
from keras.layers import SpatialDropout1D,MaxPool1D,GlobalAveragePooling1D,RepeatVector ,Add,PReLU
from keras.layers import Bidirectional, GlobalMaxPool1D,BatchNormalization,concatenate,TimeDistributed,Merge,Flatten
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.optimizers import Adam,SGD,Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Layer  
from keras import initializers, regularizers, constraints  
from keras import backend as K
from nltk.stem import SnowballStemmer
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
from criteria import *


corpus_clean= parallelize_dataframe(corpus_raw, multiply_columns_clean)
pickle.dump(corpus_clean,open("tmp_noWordNet_clean.pkl", "wb"))
corpus_twitter=pickle.load(open("tmp_noWordNet_clean.pkl", "rb")) 




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
f = codecs.open('crawl-300d-2M.vec', encoding='utf-8')
from tqdm import tqdm
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))





print('preparing embedding matrix...')
words_not_found = []
nb_words = min(max_features, len(tokenizer.word_index))
print('number with words...'+str(nb_words))

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


import tensorflow as tf

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#set up keras session

tf.keras.backend.set_session(sess)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


from numpy.random import seed
seed(1)



#default parameter are represented by my own setting of model parameter


def DPCNN(num_block=6,ngram=4,drop_ratio=0.15,last_drop_ratio=0.5):
 
    main_input=Input(shape=(maxlen,))
    embedded_sequences= Embedding(max_features, embed_size,weights=[embedding_matrix],trainable=False)(main_input)
    embedded_sequences=SpatialDropout1D(0.22)(embedded_sequences)                    
    
    
  
    
    assert num_block > 1
    
    X_shortcut1 = embedded_sequences

    x= Conv1D(filters=hidden_dim,padding='same',kernel_size=ngram)(embedded_sequences)
    x= BatchNormalization()(x)
    x = Dropout(drop_ratio)(x)
    x= PReLU()(x)
    x= Conv1D(filters=hidden_dim,padding='same', kernel_size=ngram)(x)
    x= BatchNormalization()(x)
    x = Dropout(drop_ratio)(x)
    x= PReLU()(x)
       
    
    embedding_reshape=Conv1D(nb_filter=hidden_dim,kernel_size=1,padding='same',activation='linear')(X_shortcut1)
    # connect shortcut to the main path
    embedding_reshape = PReLU()(embedding_reshape)  # pre activation
    x = Add()([embedding_reshape,x])

    x = MaxPool1D(pool_size=4, strides=2, padding='valid')(x)


  
    for i in range(2,num_block):
        X_shortcut = x

        x = Conv1D(filters=hidden_dim,padding='same', kernel_size=ngram)(x)
        x= BatchNormalization()(x)
        x = Dropout(drop_ratio)(x)
        x = PReLU()(x)
        x = Conv1D(filters=hidden_dim,padding='same', kernel_size=ngram)(x)
        x= BatchNormalization()(x)
        x = Dropout(drop_ratio)(x)
        x = PReLU()(x)

        x = Add()([X_shortcut,x])
        x = MaxPool1D(pool_size=4,strides=2, padding='valid')(x)

    
    X_shortcut_final=x 
    x = Conv1D(filters=hidden_dim,padding='same', kernel_size=ngram)(x)
    x= BatchNormalization()(x)
    x = Dropout(drop_ratio)(x)
    x = PReLU()(x)
    x = Conv1D(filters=hidden_dim,padding='same', kernel_size=ngram)(x)
    x= BatchNormalization()(x)
    x = Dropout(drop_ratio)(x)
    x = PReLU()(x) 
    
    x = Add()([X_shortcut_final,x])
    
    x = GlobalMaxPool1D()(x)
    
    x = Dense(dense_filter, activation='linear')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    
    x = Add()([X_shortcut6,x])
    
    x = GlobalMaxPool1D()(x)
    
    x = Dense(256, activation='linear')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(last_drop_ratio)(x)
    
    
    x= Dense(6, activation="sigmoid",kernel_regularizer=regularizers.l2(1e-8))(x)

    
    
    model = Model(inputs=main_input, outputs=x)
    
    nadam=Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.0022)
    model.compile(loss='binary_crossentropy',
                      optimizer=nadam,
                      metrics=['accuracy',f1_score,auc])
    print(model.summary())
    return model


batch_size = 400
#total average roc_auc: 0.9865626176353365
