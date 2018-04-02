import re
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())


import pandas as pd, numpy as np


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')



merge=pd.concat([train,test])
df=merge.reset_index(drop=True)


#Assume using glove.840B.300d=======================================
import os, re, csv, math, codecs
print('loading word embeddings...')
embeddings_index = {}
#ranking index 
#from https://nlp.stanford.edu/pubs/glove.pdf the glove is use modeled as a power-law function of the frequency rank of that word pair
f = codecs.open('glove.840B.300d.txt', encoding='utf-8')
from tqdm import tqdm
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))


words = embeddings_index

w_rank = {}
for i,word in enumerate(words):
    w_rank[word] = i

WORDS=w_rank

print("load ")

import pickle


corpus=pickle.load(open("tmp_clean.pkl", "rb")) 


def P(word):    #part from CPMP
    "Probability of `word`."
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
    return - WORDS.get(word, 0)
#=======================================================================


#Assume not using glove.840B.300d=======================================
#origin, suitable for any others, like fastext wiki
import os, re, csv, math, codecs
print('loading word embeddings...')
embeddings_index = {}
f = codecs.open('...', encoding='utf-8')

from tqdm import tqdm
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))

WORDS=embeddings_index


sum_of_words=len(WORDS)
def P(word, N=sum_of_words): 
    "Probability of `word`."
    return WORDS[word] / N
#=======================================================================


def correction(word,lower=4,upper=10): 
    "Most probable spelling correction for word."
    length=len(word)
    if length<lower or length>upper:
       return word
    elif word in WORDS:
       return word
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))



from nltk.tokenize import TweetTokenizer



tokenizer=TweetTokenizer()


import re

regx = re.compile('[a-z]+')
record = set()
def clean_correction(comment):
    comment=comment.lower()
    words=tokenizer.tokenize(comment)
    _init_=[]
    for w in words:
         #Only correct the characters and the new word
        if bool(regx.match(w)) and w not in record:
            w=correction(w)
            _init_.append(w)
            """ #save space version
            if w not in WORDS:   
                 record.add(w)
            """
                #quick version
                record.add(w)
        else:
            _init_.append(w)
    words=_init_
    sent=" ".join(words)
    return(sent)



from multiprocessing import Pool

num_partitions = 12 #number of partitions to split dataframe #4
num_cores = 4 #number of cores on your machine


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def multiply_columns_correction(data):
    data = data.apply(lambda x: clean_correction(x))
    return data

print("=======start correction=====")
       
       
import time

start=time.time()



corpus= parallelize_dataframe(corpus,multiply_columns_correction)


end=time.time()

timeStep=end-start

print("spend sencond: "+str(timeStep))


pickle.dump(corpus,open("tmp_correction_glove.pkl", "wb"))
