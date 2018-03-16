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



def multiply_columns_lemmatize_sentence(data):
    data=data.apply(lambda x:lemmatize_sentence(x))
    return data
    
#parallelize 4 core nearly 13 minutes, and don't need to do it every time, just run it at once, except doing different preprocess
corpus= parallelize_dataframe(corpus, multiply_columns_lemmatize_sentence)
#store it back to the disk
pickle.dump(corpus,open("tmp.pkl", "wb"))
corpus=pickle.load(open("tmp.pkl", "rb")) 