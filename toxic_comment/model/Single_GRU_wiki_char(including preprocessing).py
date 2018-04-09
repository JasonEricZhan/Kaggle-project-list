# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



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
from criteria import *

embed_size = 300 
max_features = 160000 
maxlen=180



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


merge=pd.concat([train,test])
df=merge.reset_index(drop=True)
corpus_raw=df.comment_text








APPO = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"
}



repl = {
    "&lt;3": " good ",
    ":d": " good ",
    ":dd": " good ",
    ":p": " good ",
    "8)": " good ",
    ":-)": " good ",
    ":)": " good ",
    ";)": " good ",
    "(-:": " good ",
    "(:": " good ",
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " bad ",
    ":(": " bad ",
    ":s": " bad ",
    ":-s": " bad ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha"}



bad_wordBank={
'fage':"shove your balls up your own ass or the ass of another to stretch your scrotum skin",
}





print("....start....cleaning")

from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']


from nltk.tokenize import TweetTokenizer


tokenizer=TweetTokenizer()


re_tok = re.compile(r'([1234567890!@#$%^&*_+-=,./<>?;:"[][}]"\'\\|�鎿�𤲞阬威鄞捍朝溘甄蝓壇螞¯岑�''\t])')





df['count_sent']=df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)
df['count_word']=df["comment_text"].apply(lambda x: len(str(x).split()))

df['avg_sent_length']=df['count_word']/df['count_sent']
print(df['count_sent'].describe())
print(df['count_word'].describe())
print(df['avg_sent_length'].describe())







from nltk.tokenize import TweetTokenizer


tokenizer=TweetTokenizer()

def clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    #Convert to lower case , so that Hi and hi are the same
    comment=comment.lower()
    #remove \n
    comment=re.sub(r"\n",".",comment)
    comment=re.sub(r"\\n\n",".",comment)
    comment=re.sub(r"fucksex","fuck sex",comment)
    comment=re.sub(r"f u c k","fuck",comment)
    comment=re.sub(r"幹","fuck",comment)
    #text = re.sub("www.* ", "<URL>", text)
    comment=re.sub(r"死","die",comment)
    comment=re.sub(r"他妈的","fuck",comment)
    comment=re.sub(r"去你妈的","fuck off",comment)
    comment=re.sub(r"肏你妈","fuck your mother",comment)
    comment=re.sub(r"肏你祖宗十八代","your ancestors to the 18th generation",comment)
    # remove leaky elements like ip,user
    comment=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)
    #removing usernames
    comment=re.sub("\[\[.*\]","",comment)
    comment = re.sub(r"you ' re", "you are", comment)
    comment = re.sub(r"wtf","what the fuck", comment)
    comment = re.sub(r"i ' m", "I am", comment)
    comment = re.sub(r"I", "one", comment)
    comment = re.sub(r"II", "two", comment)
    comment = re.sub(r"III", "three", comment)
    comment = re.sub(r'牛', "cow", comment)
    comment=re.sub(r"mothjer","mother",comment)
    comment=re.sub(r"g e t  r i d  o f  a l l  i  d i d  p l e a s e  j a ck a s s",
                   "get rid of all i did please jackass",comment)
    comment=re.sub(r"nazi","nazy",comment)
    comment=re.sub(r"withought","with out",comment)
    s=comment
    
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', 'zero')
    s = s.replace('1', 'one')
    s = s.replace('2', 'two')
    s = s.replace('3', 'three')
    s = s.replace('4', 'four')
    s = s.replace('5', 'five')
    s = s.replace('6', 'six')
    s = s.replace('7', 'seven')
    s = s.replace('8', 'eight')
    s = s.replace('9', 'night')
    s = s.replace('雲水','')
    
    comment=s
    comment = re_tok.sub(' ', comment)
   
    words=tokenizer.tokenize(comment)
    
   
    words=[APPO[word] if word in APPO else word for word in words]
    words=[bad_wordBank[word] if word in bad_wordBank else word for word in words]
    words=[repl[word] if word in repl else word for word in words]
    words = [w for w in words if not w in stop_words]
   
    
    
    sent=" ".join(words)
    sent = re.sub(r'([\'\"\/\-\_\--\_])',' ', sent)
    # Remove some special characters
    clean_sent= re.sub(r'([\;\|•«\n])',' ', sent)
    
    return(clean_sent)







from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

    
    
def lemmatize_all(sentence):
    wnl = WordNetLemmatizer()
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag.startswith("NN"):
            yield wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            yield wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            yield wnl.lemmatize(word, pos='a')
        elif tag.startswith('R'):
            yield wnl.lemmatize(word, pos='r')
        else:
            yield word    
    
    

def lemmatize_sentence(sentence):
    res = []
    lemmatizer = WordNetLemmatizer()
    for word, pos in pos_tag(word_tokenize(sentence)):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    res=" ".join(res)
        
    return res


import pandas as pd
import numpy as np

from multiprocessing import Pool

num_partitions = 8 #number of partitions to split dataframe
num_cores = 4 #number of cores on your machine

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def multiply_columns_clean(data):
    data = data.apply(lambda x: clean(x))
    return data

def multiply_columns_lemmatize_sentence(data):
    data=data.apply(lambda x:lemmatize_sentence(x))
    return data


    
    
def sent_len(x):
    doc=str(x).split("\n")
    count_one=0
    summation=0
    for word in doc:
        summation+=len(word)
    return summation/len(doc)




    
import time

start=time.time()
corpus= parallelize_dataframe(corpus_raw, multiply_columns_clean)
corpus= parallelize_dataframe(corpus, multiply_columns_lemmatize_sentence)

print("dump 1")



end=time.time()

timeStep=end-start

print("spend sencond: "+str(timeStep))

import  pickle 






df["comment_text"]=corpus



print(df["comment_text"])
print(df["comment_text"].isnull().sum())




print("....set..indirect..feature")



print("set ngram feature")




train_cl=df[:train.shape[0]]
test_cl=df[train.shape[0]:]



df['count_sent']=df["comment_text"].apply(lambda x: len(re.findall(" ",str(x)))+1)
df['count_word']=df["comment_text"].apply(lambda x: len(str(x).split()))

df['avg_sent_length']=df['count_word']/df['count_sent']
print(df['count_sent'].describe())
print(df['count_word'].describe())
print(df['avg_sent_length'].describe())






#===============char preprocessing================


def char_ngram(word,ngram=2):
    char_ngram_list=[word[i:i+ngram] for i in range(len(word)-ngram+1)]
    char_ngram_sent=" ".join(char_ngram_list)
    return char_ngram_sent





def multiply_columns_char_ngram(data):
    data2 = data.apply(lambda x: char_ngram(str(x),ngram=4))
    return data2



#using same data as word embedding or the performance may be worse!?


corpus_gram=df["comment_text"]
corpus_gram=parallelize_dataframe(corpus, multiply_columns_char_ngram)

df['count_sent']=df["comment_text"].apply(lambda x: len(re.findall(" ",str(x)))+1)
df['count_word']=df["comment_text"].apply(lambda x: len(str(x).split()))

df['avg_sent_length']=df['count_word']/df['count_sent']
print(df['count_sent'].describe())
print(df['count_word'].describe())
print(df['avg_sent_length'].describe())



from collections import Counter

# part from Dieter
def create_char_vocabulary(texts,min_count_chars=50):
    counter = Counter()
    for k, text in enumerate(texts):
        counter.update(text)

    raw_counts = list(counter.items())
    print('%s characters found' %len(counter))
    print('keepin characters with count > %s' % min_count_chars)
    vocab = [char_tuple[0] for char_tuple in raw_counts if char_tuple[1] > min_count_chars]
    char2index = {char:(ind+1) for ind, char in enumerate(vocab)}
    char2index[UNKNOWN_CHAR] = 0
    char2index[PAD_CHAR] = -1
    index2char = {ind:char for char, ind in char2index.items()}
    print('%s remaining characters' % len(char2index))
    return char2index, index2char
    
def char2seq(texts, maxlen):
    res = np.zeros((len(texts),maxlen))
    for k,text in enumerate(texts):
        seq = np.zeros((len(text))) #equals padding with PAD_CHAR
        for l, char in enumerate(text):
            try:
                id = char2index[char]
                seq[l] = id
            except KeyError:
                seq[l] = char2index[UNKNOWN_CHAR]
        seq = seq[:maxlen]
        res[k][:len(seq)] = seq
    return res


UNKNOWN_CHAR = 'ⓤ'
PAD_CHAR = '℗'


char2index, index2char = create_char_vocabulary(corpus_gram.values)
sentences_train=corpus_gram.iloc[:train.shape[0]]
sentences_test=corpus_gram.iloc[train.shape[0]:]



totalNumWords = [len(one_comment) for one_comment in sentences_train]
print("X_tr_2 mean length:"+ str(np.mean(totalNumWords )))
print("X_tr_2 max length:"+ str(max(totalNumWords) ) )
print("X_tr_2 std length:"+ str(np.std(totalNumWords )))

totalNumWords = [len(one_comment) for one_comment in sentences_test]
print("X_te_2 mean length:"+ str(np.mean(totalNumWords )))
print("X_te_2 max length:"+ str(max(totalNumWords) ) )
print("X_te_2 std length:"+ str(np.std(totalNumWords )))

maxlen_char=720  #540

X_tr_2 = char2seq(sentences_train,maxlen_char)
X_te_2 = char2seq(sentences_test,maxlen_char)



#tricky way to get the preprocessed data for training word2vec model
train["comment_text"]=train_cl["comment_text"].iloc[:train.shape[0]]

char_toxic=train.loc[train["clean"]==0,"char"]


#===========char embedding training================

from gensim.models import Word2Vec
    
model= Word2Vec(sentences=char_toxic, size=50, window=100, min_count=50, workers=2000, sg=0)  

model.save('mymodel_toxic')


weights_char=model.wv.syn0

np.save(open("self_train_weight_toxic.npz", 'wb'), weights_char)


#weights_char=np.load(open("self_train_weight_toxic.npz", 'rb'))





#===============tokenize================

print("....start....tokenizer")

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_tr = train_cl[list_classes].values      

    


list_sentences_train=train_cl.comment_text
list_sentences_test=test_cl.comment_text



print("....start....pretrain")

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



print(" maxlen is:"+str(maxlen))

print("number of different word:"+ str(len(tokenizer.word_index.items())))

if len(tokenizer.word_index.items()) < max_features:     
       max_features=len(tokenizer.word_index.items())
      

from keras.preprocessing import sequence
print('Pad sequences (samples x time)')


maxlen=180
X_tr = pad_sequences(list_tokenized_train, maxlen=maxlen,padding='post')
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen,padding='post')



print('x_train shape:', X_tr.shape)
print('x_test shape:', X_te.shape)

print("================")

print(X_tr)
print("================")
print(X_te)

print("================")

#print(X_tr_Ngram)
#print(X_te_Ngram)


from bs4 import BeautifulSoup


data=pd.concat([list_sentences_train,list_sentences_test])




X_tr_1=X_tr
X_te_1=X_te

print('x_train_1 new shape:', X_tr_1.shape)
print('x_test_1 new shape:', X_te_1.shape)




X_tr_2=X_tr_Ngram
X_te_2=X_te_Ngram

print('x_train_2 new shape:', X_tr_2.shape)
print('x_test_2 new shape:', X_te_2.shape)


#=========start to load word embedding===========


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



def bigru_pool_model_multi_input(hidden_dim_1=136,hidden_dim_2=50):
    main_input=Input(shape=(maxlen,),name='main_input')#, name='main_input'
    Ngram_input= Input(shape=(maxlen_char,), name='aux_input')#, name='aux_input'
    embedded_sequences= Embedding(max_features, embed_size,weights=[embedding_matrix],trainable=False)(main_input)
    embedded_sequences_2= Embedding(weights_char.shape[0], 50,weights=[weights_char],trainable=True)(Ngram_input)
    
    #word level
    x=SpatialDropout1D(0.22)(embedded_sequences)                    #0.1
    x_gru_1 = Bidirectional(CuDNNGRU(hidden_dim_1,recurrent_regularizer=regularizers.l2(1e-6),return_sequences=True))(x)
    
    #char level
    x_2=SpatialDropout1D(0.21)(embedded_sequences_2)                    #0.1
    x_gru_2 = Bidirectional(CuDNNGRU(hidden_dim_2,recurrent_regularizer=regularizers.l2(1e-8),return_sequences=True))(x_2)

    
    x_ave_1=GlobalAveragePooling1D()(x_gru_1)
    x_ave_2=GlobalAveragePooling1D()(x_gru_2)
    x_ave= concatenate([x_ave_1,x_ave_2])
    x_max_1=GlobalMaxPool1D()(x_gru_1)
    x_max_2=GlobalMaxPool1D()(x_gru_2)
    x_max= concatenate([x_max_1,x_max_2])
    x_dense= concatenate([x_max,x_ave])
    x_dense=BatchNormalization()(x_dense)
    x_dense= Dropout(0.35)(x_dense)
    x_dense = Dense(256, activation="elu")(x_dense)
    x_dense = Dropout(0.3)(x_dense)
    x_dense = Dense(128, activation="elu")(x_dense)
    x = Dropout(0.2)(x_dense)
    x= Dense(6, activation="sigmoid",kernel_regularizer=regularizers.l2(1e-8))(x)
    
    model = Model(inputs=[main_input,Ngram_input], outputs=x)
    nadam=Nadam(lr=0.00262, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.00325)
    model.compile(loss='binary_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy',f1_score,auc])
    print(model.summary())
    return model


batch_size = 1280   #faster for char level embedding model

#total average roc_auc: 0.9888378030202132


