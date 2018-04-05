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
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')

embed_size = 300 # how big is each word vector
max_features = 160000 # how many unique words to use (i.e num rows in embedding vector)


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')






class_names = list(train)[-6:]
multarray = np.array([100000, 10000, 1000, 100, 10, 1])
y_multi = np.sum(train[class_names].values * multarray, axis=1)

print(class_names)

print(y_multi)


from sklearn.model_selection import StratifiedKFold
splits = 10
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

# produce two lists of ids. each list has n items where n is the 
#    number of folds and each item is a pandas series of indexed id numbers
train_ids = [] 
val_ids = []
for i, (train_idx, val_idx) in enumerate(skf.split(np.zeros(train.shape[0]), y_multi)):
    train_ids.append(train.loc[train_idx, 'id'])
    val_ids.append(train.loc[val_idx, 'id'])


from sklearn.model_selection import train_test_split

x=train.iloc[:,2:].sum()
#marking comments without any tags as "clean"
rowsums=train.iloc[:,2:].sum(axis=1)
train['clean']=(rowsums==0)



import string

from nltk.corpus import stopwords

eng_stopwords = set(stopwords.words("english"))



merge=pd.concat([train,test])
df=merge.reset_index(drop=True)


merge["comment_text"]=merge["comment_text"].fillna("_na_").values


import pickle  
#pickle.dump(df, open("tmp_df.pkl", "wb")) 


#df=pickle.load(open("tmp_df.pkl", "rb")) 



from gensim.models.wrappers import FastText

import time

start=time.time()

corpus_raw=df['comment_text'].copy()





end=time.time()

timeStep=end-start

print("spend sencond: "+str(timeStep))


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


import goslate
gs = goslate.Goslate()






df['count_sent']=df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)
df['count_word']=df["comment_text"].apply(lambda x: len(str(x).split()))

df['avg_sent_length']=df['count_word']/df['count_sent']
print(df['count_sent'].describe())
print(df['count_word'].describe())
print(df['avg_sent_length'].describe())



def glove_preprocess(text):
    """
    adapted from https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

    """
    # Different regex parts for smiley faces
    eyes = "[8:=;]"
    nose = "['`\-]?"
    text = re.sub("https?://.* ", "<URL>", text)
    text = re.sub("www.* ", "<URL>", text)
    text = re.sub("/", " / ", text)
    text = re.sub("\[\[User(.*)\|", '<USER>', text)
    text = re.sub("<3", '<HEART>', text)
    text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
    text = re.sub(eyes + nose + "[Dd)]", '<SMILE>', text)
    text = re.sub("[(d]" + nose + eyes, '<SMILE>', text)
    text = re.sub(eyes + nose + "p", '<LOLFACE>', text)
    text = re.sub(eyes + nose + "\(", '<SADFACE>', text)
    text = re.sub("\)" + nose + eyes, '<SADFACE>', text)
    text = re.sub(eyes + nose + "[/|l*]", '<NEUTRALFACE>', text)
    text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
    text = re.sub("([!]){2,}", "! <REPEAT>", text)
    text = re.sub("([?]){2,}", "? <REPEAT>", text)
    text = re.sub("([.]){2,}", ". <REPEAT>", text)
    text = re.sub("(.)\1{2,}", "\1\1\1 <ELONG>", text)
    pattern = re.compile(r"(.)\1{2,}")
    text = pattern.sub(r"\1" + " <ELONG>", text)

    return text




from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

"""
import os, re, csv, math, codecs
print('loading word embeddings...')
embeddings_index = {}
#f = codecs.open('crawl-300d-2M.vec', encoding='utf-8')
#f = codecs.open('wiki.en.vec', encoding='utf-8')
f = codecs.open('glove.840B.300d.txt', encoding='utf-8')
from tqdm import tqdm
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))

print('found %s word vectors' % embeddings_index.values)


print('start to compute std, mean')


words = embeddings_index

w_rank = {}
for i,word in enumerate(words):
    w_rank[word] = i
"""
#WORDS = w_rank








import re
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())

def P(word): 
    "Probability of `word`."
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
    return - WORDS.get(word, 0)

def correction(word,limit=10): 
    "Most probable spelling correction for word."
    if len(word)>limit:
       return word
    if word in WORDS:
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
    #comment = re.sub(r"what's", "", comment)
    #comment = re.sub(r"What's", "", comment)
    #comment = re.sub(r"\'s", " ", comment)
    comment = re.sub(r"\'ve", " have ", comment)
    #comment = re.sub(r"can't", "cannot ", comment)
    comment = re.sub(r"n't", " not ", comment)
    #comment = re.sub(r"I'm", "I am", comment)
    #comment = re.sub(r" m ", " am ", comment)
    #comment = re.sub(r"\'re", " are ", comment)
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
    #comment = re.sub(r"abt", "about", comment)
    comment=re.sub(r"mothjer","mother",comment)
    comment=re.sub(r"g e t  r i d  o f  a l l  i  d i d  p l e a s e  j a ck a s s",
                   "get rid of all i did please jackass",comment)
    comment=re.sub(r"nazi","nazy",comment)
    comment=re.sub(r"withought","with out",comment)
    s=comment
    #s = re.sub(r'([\'\"\.\!\?\/\,])', r' \1 ', s)
    # Remove some special characters
    #s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Replace numbers and symbols with language #is worse
    
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
    #token='1234567890!@#$%^&*_+-=,./<>?;:"[][}]"\'\\|'

    #comment=comment.replace(token," ")
    #sub is replace in python
    
    
    #Split the sentences into words
    words=tokenizer.tokenize(comment)
    
    #print(words)
    # (')aphostophe  replacement (ie)   you're --> you are  
    # ( basic dictionary lookup : master dictionary present in a hidden block of code)
    words=[APPO[word] if word in APPO else word for word in words]
    words=[bad_wordBank[word] if word in bad_wordBank else word for word in words]
    words=[repl[word] if word in repl else word for word in words]
    #print(words)
    words=[lem.lemmatize(word, "v") for word in words]
    #print(words)
    words = [w for w in words if not w in stop_words]
    #print(words)
    #
    
    
    sent=" ".join(words)
    # remove any non alphanum,digit character
    #clean_sent=re.sub("\W+"," ",clean_sent)
    #clean_sent=re.sub("  "," ",clean_sent)
    #sent = re.sub(r'([\'\"\.\!\?\/\-\_\,])',' ', sent)
    sent = re.sub(r'([\'\"\/\-\_\--\_])',' ', sent)
    # Remove some special characters
    clean_sent= re.sub(r'([\;\|•«\n])',' ', sent)
    
    return(clean_sent)


def clean_char(comment):
    comment=comment.lower()
    comment=re.sub(r"\n",".",comment)
    comment=re.sub(r"\\n\n",".",comment)
    comment=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)
    comment=re.sub("\[\[.*\]","",comment)
    #comment = re_tok.sub(' ', comment)
    words=tokenizer.tokenize(comment)
    words=[APPO[word] if word in APPO else word for word in words]
    words = [w for w in words if not w in stop_words]
    
    sent=" ".join(words)
    #sent = re.sub(r'([\'\"\/\-\_\--\_])',' ', sent)
    #clean_sent= re.sub(r'([\;\|•«\n])',' ', sent)
    clean_sent=sent
    return(clean_sent)


def clean_correction(comment):
    words=tokenizer.tokenize(comment)
    words = [correction(w) for w in words]
    sent=" ".join(words)
    return(sent)






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

def multiply_columns_clean_And_lemmatize_sentence(data):
    data = data.apply(lambda x: clean(x))
    data=data.apply(lambda x:lemmatize_sentence(x))
    return data

def multiply_columns_lemmatize_sentence(data):
    data=data.apply(lambda x:lemmatize_sentence(x))
    return data



def multiply_columns_glove_preprocess(data):
    data = data.apply(lambda x: glove_preprocess(x))
    return data

def multiply_columns_count_sent(data):
    temp = data.apply(lambda x: sent_len(x))
    return temp
    
    
def multiply_columns_correction(data):
    data = data.apply(lambda x: clean_correction(x))
    return data
    
    
    
def multiply_columns_clean_char(data):
    data = data.apply(lambda x: clean_char(x))
    return data    
    
    
def sent_len(x):
    doc=str(x).split("\n")
    count_one=0
    summation=0
    for word in doc:
        summation+=len(word)
    return summation/len(doc)



#df['sent_length']=parallelize_dataframe(df["comment_text"], multiply_columns_clean)

#df['sent_length']=parallelize_dataframe(corpus_raw, multiply_columns_count_sent)
#print(df['sent_length'].describe())    
    
import time

start=time.time()
#corpus= parallelize_dataframe(corpus_raw, multiply_columns_clean)
#corpus= parallelize_dataframe(corpus, multiply_columns_clean_And_lemmatize_sentence)
#pickle.dump(corpus,open("tmp_noWordNet.pkl", "wb"))
#corpus_twitter= parallelize_dataframe(corpus, multiply_columns_glove_preprocess)
#corpus=pickle.load(open("tmp_noWordNet.pkl", "rb")) 
print("dump 1")



#corpus =parallelize_dataframe(corpus, multiply_columns_correction)
#pickle.dump(corpus,open("tmp_correction.pkl", "wb"))
#corpus=pickle.load(open("tmp_correction.pkl", "rb")) 
#corpus= parallelize_dataframe(corpus, multiply_columns_lemmatize_sentence)
#pickle.dump(corpus,open("tmp_clean.pkl", "wb"))
print("dump 2")
#clean_corpus=corpus.apply(lambda x :clean(x))
end=time.time()

timeStep=end-start

print("spend sencond: "+str(timeStep))

import  pickle 
#pickle.dump(corpus,open("tmp.pkl", "wb"))


corpus=pickle.load(open("tmp_clean.pkl", "rb")) 
#corpus=pickle.load(open("tmp_noWordNet.pkl", "rb")) 


clean_corpus = corpus
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


#merge['comment_text']=clean_corpus




df["comment_text"]=clean_corpus

#df["twitter_comment_text"]=corpus_twitter


def create_docs(df, n_gram_max=4):
    def add_ngram(q, n_gram_max):
            ngrams = []
            for n in range(2, n_gram_max+1):
                for w_index in range(len(q)-n+1):
                    ngrams.append('--'.join(q[w_index:w_index+n]))
            return q + ngrams
        
    docs = []
    for doc in df["comment_text"]:
#       preprocess already run
#       doc = preprocess(doc).split()        
        doc = doc.split()
        docs.append(' '.join(add_ngram(doc, n_gram_max)))
    
    return docs


print(df["comment_text"])
print(df["comment_text"].isnull().sum())
#print(type(df["comment_text"].iloc[99]))


print("....set..indirect..feature")



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


df=merge.reset_index(drop=True)



#f=open('dirtyWord_bank2.txt',"rt")




#df["dirty_word_similarity"]= parallelize_dataframe(df["comment_text"], multiply_columns_similarity)




#import cPickle as pickle 
#pickle.dump(df, open("tmp_df.pkl", "wb")) 


#df=pickle.load(open("tmp_df.pkl", "rb")) 




print("set ngram feature")




#pickle.dump(corpus,open("tmp_tweet_glove_noWordNet.pkl", "wb"))

print("dump 1-1")

#corpus=pickle.load(open("tmp_tweet_glove_noWordNet.pkl", "rb")) 

#parallelize_dataframe(corpus, multiply_columns_lemmatize_sentence)

#train_Ngram=create_docs(train_cl, n_gram_max=3)   #data type is list
#test_Ngram=create_docs(test_cl, n_gram_max=3)




#test_cl=test_cl.reset_index(drop=True)

#dtrain, dval = train_test_split(train_cl, random_state=2345, train_size=0.8)






def char_ngram(word,ngram=2):
    char_ngram_list=[word[i:i+ngram] for i in range(len(word)-ngram+1)]
    char_ngram_sent=" ".join(char_ngram_list)
    return char_ngram_sent





def multiply_columns_char_ngram(data):
    data2 = data.apply(lambda x: char_ngram(str(x),ngram=4))
    #data=data.apply(lambda x:lemmatize_sentence(x))
    return data2


train_cl=df[:train.shape[0]]
test_cl=df[train.shape[0]:]


#train_Ngram=np.array(train_Ngram)
#test_Ngram=np.array(test_Ngram)

#merge_Ngram=np.hstack(train_Ngram,test_Ngram)

#df["comment_text"].iloc[:train.shape[0]]=train_Ngram
#df["comment_text"].iloc[train.shape[0]:]=test_Ngram


df['count_sent']=df["comment_text"].apply(lambda x: len(re.findall(" ",str(x)))+1)
df['count_word']=df["comment_text"].apply(lambda x: len(str(x).split()))

df['avg_sent_length']=df['count_word']/df['count_sent']
print(df['count_sent'].describe())
print(df['count_word'].describe())
print(df['avg_sent_length'].describe())


#corpus_gram=df["comment_text"]
#corpus_gram=parallelize_dataframe(corpus_raw, multiply_columns_char_ngram)




from collections import Counter

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



#train["comment_text"]=df["comment_text"].iloc[:train.shape[0]]



#print("toxic number: "+str(train['clean'].value_counts()))

#print(len(train.loc[train['clean']==0]))

#toxic_corpus=train.loc[train['clean']==0,"comment_text"]


#print(toxic_corpus)

#corpus_clean_char=parallelize_dataframe(corpus_raw,multiply_columns_clean_char)

#char2index, index2char = create_char_vocabulary(corpus)

#char_corpus=parallelize_dataframe(corpus, multiply_columns_char_ngram)

#df["char"]=char_corpus

#print(df["char"])








#sentences_train=char_corpus.iloc[:train.shape[0]]
#sentences_test=char_corpus.iloc[train.shape[0]:]

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




train["char"]=char_corpus.iloc[:train.shape[0]]

char_toxic=train.loc[train["clean"]==0,"char"]

from gensim.models import Word2Vec
    
model= Word2Vec(sentences=char_toxic, size=50, window=100, min_count=50, workers=2000, sg=0)  

model.save('mymodel_toxic')


weights=model.wv.syn0

np.save(open("self_train_weight_toxic.npz", 'wb'), weights)

#weights_2 = np.load(open("self_train_weight.npz", 'rb'))

#weights=np.load(open("self_train_weight_NoClean.npz", 'rb'))

weights=np.load(open("self_train_weight_toxic.npz", 'rb'))

##test_char_Ngram=df[train.shape[0]:]


#test_cl=test_cl.reset_index(drop=True)

print("char gram")

#df['count_word']=df["comment_text"].apply(lambda x: len(str(x).split()))
#print(df['count_word'].describe())

"""
print("...original length of dtrain "+str(len(dtrain)))
      
      

L = len(dtrain)
df_irr = dtrain[dtrain.clean == 0]
while len(dtrain) < 2*L:
    dtrain = dtrain.append(df_irr, ignore_index=True)

     

print("...after length of dtrain "+str(len(dtrain)))
      


    
print("...original length of dval "+str(len(dval)))
      
      

L = len(dval)
df_irr = dval[dval.clean == 0]
while len(dval) < 2*L:
    dval = dval.append(df_irr, ignore_index=True)


     
        

print("...after length of dval "+str(len(dval)))

    

"""    
    
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

#maxlen=400
maxlen=180#int(np.mean(totalNumWords )+np.std(totalNumWords )*2+1)

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




X_tr_1=X_tr#.reshape((-1,MAX_SENTS,MAX_SENT_LENGTH))
X_te_1=X_te#.reshape((-1,MAX_SENTS,MAX_SENT_LENGTH))

print('x_train_1 new shape:', X_tr_1.shape)
print('x_test_1 new shape:', X_te_1.shape)




X_tr_2=X_tr_Ngram
X_te_2=X_te_Ngram

print('x_train_2 new shape:', X_tr_2.shape)
print('x_test_2 new shape:', X_te_2.shape)



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



def get_model():
    main_input=Input(shape=(maxlen,),name='main_input')#, name='main_input'
    Ngram_input= Input(shape=(maxlen_char,), name='aux_input')#, name='aux_input'
    embedded_sequences= Embedding(max_features, embed_size,weights=[embedding_matrix],trainable=False)(main_input)
    embedded_sequences_2= Embedding(weights.shape[0], 50,weights=[weights],trainable=True)(Ngram_input)
    
    #word level
    hidden_dim=136
    x=SpatialDropout1D(0.22)(embedded_sequences)                    #0.1
    x_gru_1 = Bidirectional(CuDNNGRU(hidden_dim,recurrent_regularizer=regularizers.l2(1e-6),return_sequences=True))(x)
    
    #char level
    hidden_dim=50
    x_2=SpatialDropout1D(0.21)(embedded_sequences_2)                    #0.1
    x_gru_2 = Bidirectional(CuDNNGRU(hidden_dim,recurrent_regularizer=regularizers.l2(1e-8),return_sequences=True))(x_2)

    
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
    nadam=Nadam(lr=0.0029, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.00325)
    model.compile(loss='binary_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy',f1_score,auc])
    print(model.summary())
    return model


batch_size = 1600   #faster for char level embedding model

#total average roc_auc: 0.9888378030202132


