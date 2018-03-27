from __future__ import absolute_import, division
import sys, os, re, csv, codecs, numpy as np, pandas as pd


from keras.preprocessing.sequence import pad_sequences
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english') 



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


from sklearn.model_selection import train_test_split

x=train.iloc[:,2:].sum()
rowsums=train.iloc[:,2:].sum(axis=1)
train['clean']=(rowsums==0)



merge=pd.concat([train,test])
df=merge.reset_index(drop=True)


merge["comment_text"]=merge["comment_text"].fillna("_na_").values

corpus_raw=df.comment_text


no_abbre = {
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



emoji = {
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


#Can add more to do the extension
bad_wordBank={
'fage':"shove your balls up your own ass or the ass of another to stretch your scrotum skin",
}


print("....start....cleaning")



#=================stop word=====================


from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to']

import string

from nltk.corpus import stopwords

eng_stopwords = set(stopwords.words("english"))


#=================other special code=====================

re_tok = re.compile(r'([�鎿�𤲞阬威鄞捍朝溘甄蝓壇螞¯岑�''\t])')


#=================replace the duplicate word=====================
import re

def substitute_repeats_fixed_len(text, nchars, ntimes=4):
        # Find substrings that consist of `nchars` non-space characters
        # and that are repeated at least `ntimes` consecutive times,
        # and replace them with a single occurrence.
        # Examples: 
        # abbcccddddeeeee -> abcde (nchars = 1, ntimes = 2)
        # abbcccddddeeeee -> abbcde (nchars = 1, ntimes = 3)
        # abababcccababab -> abcccab (nchars = 2, ntimes = 2)
        return re.sub(r"(\S{{{}}})(\1{{{},}})".format(nchars, ntimes-1),
                      r"\1", text)

def substitute_repeats(text, ntimes=4):
        # Truncate consecutive repeats of short strings
        for nchars in range(1, 20):
            text = substitute_repeats_fixed_len(text, nchars, ntimes)
        return text



#=================choose one of tokenizer=======================
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

from nltk.tokenize import TweetTokenizer

tokenizer=TweetTokenizer()


#=================final clean function=======================
def clean(comment):
    comment=comment.lower()
    #remove \n
    comment=re.sub(r"\n",".",comment)
    comment=re.sub(r"\\n\n",".",comment)
    comment=re.sub(r"fucksex","fuck sex",comment)
    comment=re.sub(r"f u c k","fuck",comment)
    comment=re.sub(r"幹","fuck",comment)
    #Chinese bad word
    comment=re.sub(r"死","die",comment)
    comment=re.sub(r"他妈的","fuck",comment)
    comment=re.sub(r"去你妈的","fuck off",comment)
    comment=re.sub(r"肏你妈","fuck your mother",comment)
    comment=re.sub(r"肏你祖宗十八代","your ancestors to the 18th generation",comment)
    # remove leaky elements like ip,user
    comment=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)
    #removing usernames
    comment=re.sub("\[\[.*\]","",comment)
    comment = re.sub(r"\'ve", " have ", comment)
    
    comment = re.sub(r"n't", " not ", comment)
    comment = re.sub(r"\'d", " would ", comment)
    comment = re.sub(r"\'ll", " will ", comment)
    comment = re.sub(r"ca not", "cannot", comment)
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
    comment=substitute_repeats(comment)
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
    
    words=[no_abbre[word] if word in APPO else word for word in words]
    words=[bad_wordBank[word] if word in bad_wordBank else word for word in words]
    words=[emoji[word] if word in repl else word for word in words]
    words=[lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if not w in stop_words]
    
    
    sent=" ".join(words)
    # Remove some special characters, or noise charater, but do not remove all!!
    sent = re.sub(r'([\'\"\/\-\_\--\_])',' ', sent)
    clean_sent= re.sub(r'([\;\|•«\n])',' ', sent)
    
    return(clean_sent)



#==================set up the multi core function to do preprocessin=======================

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

    
