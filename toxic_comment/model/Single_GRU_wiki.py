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
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')

embed_size = 300 
max_features = 160000
maxlen=180


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
merge=pd.concat([train,test])
df=merge.reset_index(drop=True)



corpus_raw=df.comment_text


from commen_preprocess import *
from word_net_lemmatize import *
from criteria import *



import time

start=time.time()
corpus_pre1= parallelize_dataframe(corpus_raw, multiply_columns_clean)
corpus_lemmatize= parallelize_dataframe(corpus_pre1, multiply_columns_lemmatize_sentence)
pickle.dump(corpus_lemmatize,open("tmp_WordNet_corpus_lemmatize.pkl", "wb"))
corpus_lemmatize=pickle.load(open("tmp_WordNet_corpus_lemmatize.pkl", "rb")) 

end=time.time()

timeStep=end-start

print("spend sencond: "+str(timeStep))





df["comment_text"]=corpus_lemmatize



train_cl=df[:train.shape[0]]
test_cl=df[train.shape[0]:]
    
print("....start....tokenizer")

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_tr = train_cl[list_classes].values      

    


list_sentences_train=train_cl.comment_text
list_sentences_test=test_cl.comment_text


from numpy import asarray
from numpy import zeros


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



print("number of different word:"+ str(len(tokenizer.word_index.items())))

if len(tokenizer.word_index.items()) < max_features:     
       max_features=len(tokenizer.word_index.items())

from keras.preprocessing import sequence
print('Pad sequences (samples x time)')




X_tr = pad_sequences(list_tokenized_train, maxlen=maxlen,padding='post')
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen,padding='post')




print('x_train shape:', X_tr.shape)
print('x_test shape:', X_te.shape)

print("================")

print(X_tr)
print("================")
print(X_te)

print("================")






from gensim.models.wrappers import FastText

print("start...loading...wiki....en")

model = FastText.load_fasttext_format('wiki.en')


nb_words= min(max_features, len(tokenizer.word_index))

embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in tokenizer.word_index.items():
    if i >= nb_words:
        continue
    if word in model.wv:
        embedding_matrix[i] = model[word]
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))




import tensorflow as tf

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#set up keras session

tf.keras.backend.set_session(sess)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


from numpy.random import seed
seed(1)



def bigru_pool_model():
    main_input=Input(shape=(maxlen,),name='main_input')#, name='main_input'
    embedded_sequences= Embedding(max_features, embed_size,weights=[embedding_matrix],trainable=trainable)(main_input)

    hidden_dim=136
    x=SpatialDropout1D(0.22)(embedded_sequences)                    #0.1
    x_gru_1 = Bidirectional(CuDNNGRU(hidden_dim,recurrent_regularizer=regularizers.l2(1e-6),return_sequences=True))(x)
    x_ave=GlobalAveragePooling1D()(x_gru_1)
    x_max=GlobalMaxPool1D()(x_gru_1)
    x_dense= concatenate([x_max,x_ave])
    x_dense=BatchNormalization()(x_dense)
    x_dense= Dropout(0.35)(x_dense)
    x_dense = Dense(256, activation="elu")(x_dense)
    x_dense = Dropout(0.3)(x_dense)
    x_dense = Dense(128, activation="elu")(x_dense)
    x = Dropout(0.2)(x_dense)
    x= Dense(6, activation="sigmoid",kernel_regularizer=regularizers.l2(1e-8))(x)
 
    
    
    
    model = Model(inputs=main_input, outputs=x)
    nadam=Nadam(lr=0.00225, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.00325)
    model.compile(loss='binary_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy',f1_score,auc])
    print(model.summary())
    return model




batch_size = 640

#total average roc_auc: 0.9891360615629836
