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

    
 #if you want to know what the tag means, please see there https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html   

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


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def multiply_columns_lemmatize_sentence(data):
    data=data.apply(lambda x:lemmatize_sentence(x))
    return data
    
#parallelize 4 core nearly 13 minutes, and don't need to do it every time, just run it at once, except doing different preprocess
corpus= parallelize_dataframe(corpus, multiply_columns_lemmatize_sentence)
#store it back to the disk
pickle.dump(corpus,open("tmpWordNetlem.pkl", "wb"))
corpus=pickle.load(open("tmpWordNetlem.pkl", "rb")) 
