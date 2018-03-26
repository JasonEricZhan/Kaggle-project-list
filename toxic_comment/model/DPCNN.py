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
stemmer = SnowballStemmer('english')

embed_size = 300 # how big is each word vector
max_features = 160000 # how many unique words to use (i.e num rows in embedding vector)
#maxlen = 300 # max number of words in a comment to use
maxlen=360

#EMBEDDING_FILE="glove.6B.300d.txt"

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


#train = train.sample(frac=1)

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

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

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
    comment=substitute_repeats(comment)
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

def createStemer(text):
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)    
    
    # Return a list of words
    return(text)




from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    #elif treebank_tag.startswith('J'):
        #return wordnet.ADJ
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
    
    
def sent_len(x):
    doc=str(x).split("\n")
    count_one=0
    summation=0
    for word in doc:
        summation+=len(word)
    return summation/len(doc)



#df['sent_length']=parallelize_dataframe(df["comment_text"], multiply_columns_clean)

df['sent_length']=parallelize_dataframe(corpus_raw, multiply_columns_count_sent)
print(df['sent_length'].describe())    
    
import time

start=time.time()
#corpus= parallelize_dataframe(corpus_raw, multiply_columns_clean)
#corpus= parallelize_dataframe(corpus, multiply_columns_clean_And_lemmatize_sentence)
#pickle.dump(corpus,open("tmp_noWordNet.pkl", "wb"))
#corpus_twitter= parallelize_dataframe(corpus, multiply_columns_glove_preprocess)
corpus=pickle.load(open("tmp_noWordNet.pkl", "rb")) 
print("dump 1")

#corpus= parallelize_dataframe(corpus, multiply_columns_lemmatize_sentence)
#pickle.dump(corpus,open("tmp.pkl", "wb"))
print("dump 2")
#clean_corpus=corpus.apply(lambda x :clean(x))
end=time.time()

timeStep=end-start

print("spend sencond: "+str(timeStep))

import  pickle 
#pickle.dump(corpus,open("tmp.pkl", "wb"))


#corpus=pickle.load(open("tmp.pkl", "rb")) 
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

"""
import re
df['count_sent']=df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)
#Word count in each comment:
df['count_word']=df["comment_text"].apply(lambda x: len(str(x).split()))
#Unique word count
df['count_unique_word']=df["comment_text"].apply(lambda x: len(set(str(x).split())))
#Letter count
df['count_letters']=df["comment_text"].apply(lambda x: len(str(x)))
#punctuation count 
df["count_punctuations"] =df["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
#upper case words count
df["count_words_upper"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
#title case words count
df["count_words_title"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
#Number of stopwords
df["count_stopwords"] = df["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
#Average length of the words
df["mean_word_len"] = df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

df["mean_word_len"] =df["mean_word_len"].fillna(0)


df['word_unique_percent']=df['count_unique_word']*100/(df['count_word']+1)
#derived features
#Punct percent in each comment:
df['punct_percent']=df['count_punctuations']*100/(df['count_word']+1)

df["count_words_lower"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.islower()]))





print("....set..dirtyWBank")
"""

#f=open('dirtyWord_bank2.txt',"rt")
"""
with open('dirtyWord_bank.txt',"rt") as f:
     dirtyWBank =f.readlines()
with open('dirtyWord_bank2.txt',"rt") as f:
     dirtyWBank2 =f.readlines()

dirtyWBank = [x.strip() for x in dirtyWBank] 

dirtyWBank2 = [x.strip() for x in dirtyWBank2] 

dirty_set=set(dirtyWBank+dirtyWBank2)

"""

def findListOfword(word_,list_):
    count=0
    for i in list_:
               #if i == j:
        temp=re.findall(i,word_)
        if temp==[i]:
           count+=1
           
    return count
"""

"""
from nltk.corpus import wordnet
from itertools import product

def MeanSimilarity(list1,list2):
    allsyns1 = set(ss for word in list1 for ss in wordnet.synsets(word))
    allsyns2 = set(ss for word in list2 for ss in wordnet.synsets(word))
    storeDate=[(wordnet.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2)]
    summation=0
    for i in range(0,len(storeDate)):
        summation+=storeDate[i][0]
    return (summation/len(storeDate))*(1/len(list1))*(len(set(list1)))
 
    

def multiply_columns_word_freq_count(data):
    data2 = data.apply(lambda x: findListOfword(x,dirty_set))
    #data=data.apply(lambda x:lemmatize_sentence(x))
    return data2

#df["dirty_word_freq_count"]=parallelize_dataframe(df["comment_text"], multiply_columns_word_freq_count)


def multiply_columns_similarity(data):
    data2 = data.apply(lambda x: MeanSimilarity(x.split(),dirtyWBank))
    #data=data.apply(lambda x:lemmatize_sentence(x))
    return data2

"""
print("...extract..last..word..")


def last_word_sub(string_,windowlen):  
    if(len(string_)>=windowlen):
        new_string_=string_[-windowlen:]
    else:
        new_string_=string_
    return new_string_
    
def last_word(data):
    data['comment_text']=data['comment_text'].apply(lambda x: last_word_sub(x,maxlen))
    return data
    
    
df=parallelize_dataframe(df,last_word)
"""



#df["dirty_word_similarity"]= parallelize_dataframe(df["comment_text"], multiply_columns_similarity)




#import cPickle as pickle 
#pickle.dump(df, open("tmp_df.pkl", "wb")) 


#df=pickle.load(open("tmp_df.pkl", "rb")) 

"""
print("....start....logistic")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

vectorizer = TfidfVectorizer(max_features=80000)
X = vectorizer.fit_transform(merge['comment_text'])

col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


nrow_train = train.shape[0]
#train_cl=merge[:train.shape[0]]
#test_cl=merge[train.shape[0]:]


preds = np.zeros((test.shape[0], len(col)))



loss = []

for i, j in enumerate(col):
    print('===Fit '+j)
    model = LogisticRegression(C=2)
    model.fit(X[:nrow_train], train[j])
    preds[:,i] = model.predict_proba(X[nrow_train:])[:,1]
    
    pred_train = model.predict_proba(X[:nrow_train])[:,1]
    print('ROC AUC:', roc_auc_score(train[j], pred_train))
    loss.append(roc_auc_score(train[j], pred_train))
    
print('mean column-wise ROC AUC:', np.mean(loss))
    
subm = pd.read_csv("sample_submission.csv")
    
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)
submission.to_csv('submission.csv', index=False)

"""



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
#corpus_gram=parallelize_dataframe(corpus, multiply_columns_char_ngram)



"""

train["comment_text"]=df["comment_text"].iloc[:train.shape[0]]



print("toxic number: "+str(train['clean'].value_counts()))

print(len(train.loc[train['clean']==0]))

toxic_corpus=train.loc[train['clean']==0,"comment_text"]


print(toxic_corpus)
"""
#char_corpus=parallelize_dataframe(corpus_raw, multiply_columns_char_ngram)

#df["char"]=char_corpus

#print(df["char"])

"""
print('after char gram')

print(toxic_corpus)

df["comment_text"]=corpus_gram

print(df["comment_text"])

print(len(toxic_corpus))

from gensim.models import Word2Vec
    
#model_ted = Word2Vec(sentences=toxic_corpus, size=100, window=80, min_count=1, workers=2000, sg=0)  

#weights=model_ted.wv.syn0

#np.save(open("self_train_weight.npz", 'wb'), weights)

weights_2 = np.load(open("self_train_weight.npz", 'rb'))


"""


##test_char_Ngram=df[train.shape[0]:]


test_cl=test_cl.reset_index(drop=True)

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
#y_val=dval[list_classes].values      
    


list_sentences_train=train_cl.comment_text
#list_sentences_val=dval.comment_text
list_sentences_test=test_cl.comment_text

#list_sentences_train_Ngram=df["char"].iloc[:train.shape[0]]
#list_sentences_test_Ngram=df["char"].iloc[train.shape[0]:]

"""
list_sentences_train_twitter=corpus_twitter[:train.shape[0]]
#list_sentences_val=dval.comment_text
list_sentences_test_twitter=corpus_twitter[train.shape[0]:]

"""

print("....start....pretrain")
"""
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std



word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
"""
from numpy import asarray
from numpy import zeros
"""
vocab_size = len(tokenizer.word_index) + 1


embeddings_index = dict()
f = open('glove.6B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

nb_words = min(max_features, len(tokenizer.word_index))
embedding_matrix = zeros((nb_words, embed_size))
for word, i in tokenizer.word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
"""
        
"""        




from keras.preprocessing import text, sequence


X_train = pad_sequences(list(map(text2sequence, train_cl.comment_text)), maxlen=maxlen,padding='post')
#X_val = sequence.pad_sequences(list(map(text2sequence, texts_val)), maxlen=100)
X_test = pad_sequences(list(map(text2sequence, test_cl.comment_text)), maxlen=maxlen,padding='post')       
       
X_tr=X_train    
X_te=X_test
#X_t=np.hstack((X_t,train[col]))

#X_te=np.hstack((X_te,test[col]))

"""


print("....At....Tokenizer")


puncuate=r'([\.\!\?\:\,])'

from keras.preprocessing.text import Tokenizer
#filters='"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
#filters='!"#$%&()*+-/:;<=>?@[\\]^_`{|}~\t\n'
tokenizer = Tokenizer(num_words=max_features,oov_token=puncuate)
tokenizer.fit_on_texts(list(list_sentences_train)+list(list_sentences_test))


"""
from keras.preprocessing.text import text_to_word_sequence
maxlen=350
MAX_SENTS=40
data = np.zeros((len(list_sentences_train), MAX_SENTS, maxlen), dtype='int32')

for i, sentences in enumerate(list_sentences_train):
    for j, sent in enumerate(sentences):
        if j< MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k=0
            for _, word in enumerate(wordTokens):
                if k< maxlen and tokenizer.word_index[word]<max_features:
                    data[i,j,k] = tokenizer.word_index[word]
                    k=k+1
"""
                    
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
#list_tokenized_train=tokenizer.texts_to_matrix(list_sentences_train,mode='tfidf')
#list_tokenized_val = tokenizer.texts_to_sequences(list_sentences_val)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)                    
                    

 
    
"""    
from keras.preprocessing.text import Tokenizer
#filters='"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
#filters='!"#$%&()*+-/:;<=>?@[\\]^_`{|}~\t\n'
tokenizer = Tokenizer(num_words=max_features,char_level=True,oov_token=puncuate)
tokenizer.fit_on_texts(list(list_sentences_train_Ngram)+list(list_sentences_test_Ngram))

list_sentences_train_Ngram= tokenizer.texts_to_sequences(list_sentences_train_Ngram)
#list_tokenized_train=tokenizer.texts_to_matrix(list_sentences_train,mode='tfidf')
#list_tokenized_val = tokenizer.texts_to_sequences(list_sentences_val)
list_sentences_test_Ngram = tokenizer.texts_to_sequences(list_sentences_test_Ngram)        
"""

    
    
totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
print("mean length:"+ str(np.mean(totalNumWords )))
print("max length:"+ str(max(totalNumWords) ) )
print("std length:"+ str(np.std(totalNumWords )))


#maxlen=180#int(np.mean(totalNumWords )+np.std(totalNumWords )*2+1)

print(" maxlen is:"+str(maxlen))

print("number of different word:"+ str(len(tokenizer.word_index.items())))

if len(tokenizer.word_index.items()) < max_features:     
       max_features=len(tokenizer.word_index.items())
      
"""
def create_ngram_set(input_list, ngram_value=2):
   
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

# Set parameters:
# ngram_range = 2 will add bi-grams features
ngram_range = 2
batch_size = 1024
embedding_dims = embed_size


print('Loading data...')
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(list_tokenized_train), 'train sequences')
print(len(list_tokenized_test), 'test sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, list_tokenized_train)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, list_tokenized_test)), dtype=int)))

if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in list_tokenized_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(list_tokenized_train, token_indice, ngram_range)
    x_test = add_ngram(list_tokenized_test, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))
"""

from keras.preprocessing import sequence
print('Pad sequences (samples x time)')
#list_tokenized_test=tokenizer.texts_to_matrix(list_sentences_test,mode='tfidf')



X_tr = pad_sequences(list_tokenized_train, maxlen=maxlen,padding='post')
#X_val=pad_sequences(list_tokenized_val, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen,padding='post')

"""
maxlen=200
X_tr_Ngram = pad_sequences(list_sentences_train_Ngram, maxlen=maxlen,padding='post')
#X_val=pad_sequences(list_tokenized_val, maxlen=maxlen)
X_te_Ngram = pad_sequences(list_sentences_test_Ngram, maxlen=maxlen,padding='post')
"""


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

"""
def clean_str(string):
    
    #Tokenization/string cleaning for dataset
    #Every dataset is lower cased except
    
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()



for idx in range(list_sentences_train.shape[0]):
    text = BeautifulSoup(list_sentences_train[idx])
    text = clean_str(text.get_text().encode('ascii','ignore'))
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    reviews.append(sentences)
    
    labels.append(list_sentences_train.sentiment[idx])
"""

MAX_SENTS=3
MAX_SENT_LENGTH=80
#data = np.zeros((len(list_sentences_train), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')


X_tr_1=X_tr#.reshape((-1,MAX_SENTS,MAX_SENT_LENGTH))
X_te_1=X_te#.reshape((-1,MAX_SENTS,MAX_SENT_LENGTH))

print('x_train_1 new shape:', X_tr_1.shape)
print('x_test_1 new shape:', X_te_1.shape)

"""
MAX_SENTS_2=1
MAX_SENT_LENGTH_2=200

X_tr_2=X_tr_Ngram#.reshape((-1,MAX_SENTS_2,MAX_SENT_LENGTH_2))
X_te_2=X_te_Ngram#.reshape((-1,MAX_SENTS_2,MAX_SENT_LENGTH_2))

print('x_train_2 new shape:', X_tr_2.shape)
print('x_test_2 new shape:', X_te_2.shape)
"""
"""
for i, sentences in enumerate(reviews):
    for j, sent in enumerate(sentences):
        if j< MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            #update 1/10/2017 - bug fixed - set max number of words
            k=0
            for _, word in enumerate(wordTokens):
                if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
                    data[i,j,k] = tokenizer.word_index[word]
                    k=k+1

"""




"""
from gensim.models import KeyedVectors



word2vec = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))


nb_words= min(max_features, len(tokenizer.word_index))

embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in tokenizer.word_index.items():
    if i >= nb_words:
        continue
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
"""

import os, re, csv, math, codecs
print('loading word embeddings...')
embeddings_index = {}
f = codecs.open('crawl-300d-2M.vec', encoding='utf-8')
#f = codecs.open('wiki.en.vec', encoding='utf-8')
#f = codecs.open('glove.840B.300d.txt', encoding='utf-8')
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

"""
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open('wiki.en.vec'))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
"""

#all_embs = np.stack(embeddings_index.values())
#emb_mean,emb_std = all_embs.mean(), all_embs.std()

#np.mean(all_embs),np.std(all_embs)

print('preparing embedding matrix...')
words_not_found = []
nb_words = min(max_features, len(tokenizer.word_index))
print('number with words...'+str(nb_words))
container_embeddings_index=list(embeddings_index.values())
#embedding_matrix = np.random.normal(np.mean(container_embeddings_index),np.std(container_embeddings_index), (nb_words, embed_size))
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



"""
import os, re, csv, math, codecs
print('loading word embeddings...')
embeddings_index = {}
f = codecs.open('glove.twitter.27B.200d.txt', encoding='utf-8')
#f = codecs.open('wiki.en.vec', encoding='utf-8')
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

print('preparing embedding matrix...')
words_not_found = []
nb_words = min(max_features, len(tokenizer.word_index))
print('number with words...'+str(nb_words))
container_embeddings_index=list(embeddings_index.values())
#embedding_matrix = np.random.normal(np.mean(container_embeddings_index),np.std(container_embeddings_index), (nb_words, embed_size))
embed_size_twitter=200
embedding_matrix_2 = np.zeros((nb_words, embed_size_twitter))
for word, i in tokenizer.word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_2[i] = embedding_vector
    else:
        words_not_found.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix_2, axis=1) == 0))
"""


"""
def dot_product(x, kernel):
    
    #Wrapper for dot product operation, in order to be compatible with both
    #Theano and Tensorflow
    #Args:
        #x (): input
        #kernel (): weights
    #Returns:
   
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

"""
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



    

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    

class AttentionWithContext(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self, init='glorot_uniform', kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None, bias_constraint=None,  **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get('glorot_uniform')

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight((input_shape[-1], 1),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.b = self.add_weight((input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)

        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.built = True

    def compute_mask(self, input, mask):
        return None

    def call(self, x, mask=None):
        # (x, 40, 300) x (300, 1)
        multData =  K.dot(x, self.kernel) # (x, 40, 1)
        multData = K.squeeze(multData, -1) # (x, 40)
        multData = multData + self.b # (x, 40) + (40,)

        multData = K.tanh(multData) # (x, 40)

        multData = multData * self.u # (x, 40) * (40, 1) => (x, 1)
        multData = K.exp(multData) # (X, 1)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            mask = K.cast(mask, K.floatx()) #(x, 40)
            multData = mask*multData #(x, 40) * (x, 40, )

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        multData /= K.cast(K.sum(multData, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        multData = K.expand_dims(multData)
        weighted_input = x * multData
        return K.sum(weighted_input, axis=1)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1],)    
    
    


col=['count_sent','count_unique_word','count_letters',"count_punctuations","count_words_upper","count_words_title","count_stopwords","mean_word_len",'word_unique_percent','punct_percent',"dirty_word_freq_count"]#"dirty_word_similarity"]



df=df.replace([np.inf, -np.inf], np.nan)

print("....start....normalize")
print(df.isnull().sum())
#print(np.all(np.isfinite(df)))

#print(df[col])

#df = df.reset_index()



def clean_dataset(df):
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
"""
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
#norm=Normalizer(norm='l2')
df[col]=clean_dataset(df[col])
norm=RobustScaler()

norm.fit(df[col].astype(np.float64).values)
df_norm=norm.transform(df[col].astype(np.float64).values)

train_after=df[:train.shape[0]]
test_after=df[train.shape[0]:]
train_after[col]=df_norm[:train.shape[0]]
test_after[col]=df_norm[train.shape[0]:]
"""

print("complete preprocess")



import sys
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))
from keras import initializers
from keras.engine import InputSpec, Layer
from keras import backend as K
class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
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

tf.keras.backend.set_session(sess)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


from numpy.random import seed
seed(1)




from keras import initializers
class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])


import keras.backend as K

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score        
    

def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)    
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)    
    return TP/P


def auc(y_true, y_pred):   
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

from keras.layers.advanced_activations import LeakyReLU, PReLU

def get_model():
    #embed_size = 128
    #1, maxlen,
    #maxlen=MAX_SENT_LENGTH
    main_input=Input(shape=(maxlen,))#, name='main_input'
    #maxlen=MAX_SENT_LENGTH_2
    #Ngram_input= Input(shape=(maxlen,), name='aux_input')#, name='aux_input'
    #sentence_input=main_input
    embedded_sequences= Embedding(max_features, embed_size,weights=[embedding_matrix],trainable=False)(main_input)
    #embedded_sequences_2= Embedding(max_features,200,trainable=True)(Ngram_input)
    
    #weights=[embedding_matrix]
    #x = Bidirectional(LSTM(128,dropout=0.2,recurrent_dropout=0.3, return_sequences=True))(x)
    #x = Bidirectional(LSTM(128,dropout=0.2,recurrent_dropout=0.3, return_sequences=True))(x)
    #x=Conv1D(filters=256, kernel_size=10, strides=1, padding='same', activation='relu')(main_input)
    #x = Dropout(0.2)(x)
    #x = Bidirectional(GRU(64,recurrent_dropout=0.2, return_sequences=True))(x)
    #x = Dropout(0.2)(x)
    #x = Bidirectional(GRU(64,recurrent_dropout=0.2, return_sequences=True))(x)
    #x = Bidirectional(GRU(32,recurrent_dropout=0.2, return_sequences=True))(x)
    #x_1 = Bidirectional(LSTM(128,activation='relu',recurrent_dropout=0.15,return_sequences=True))(x_1)
    #x_1 = Dropout(0.2)(x_1)
    
    hidden_dim=84   #300/4
    #x_1=Conv1D(filters=hidden_dim, kernel_size=6, strides=1, padding='same', activation='relu')(x_1)
    #x_1 = Dropout(0.35)(x_1)
    #x_1 = MaxPool1D(2,padding='same')(x_1)
    #x_1=Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(x_1)
    #x_1 = Dropout(0.3)(x_1)
    #x_1 = MaxPool1D(padding='same')(x_1)
    
    #embedded_sequences = concatenate([embedded_sequences,embedded_sequences_2])
    embedded_sequences=SpatialDropout1D(0.21)(embedded_sequences)                    #0.1
    #x_gru_1 = Bidirectional(CuDNNGRU(hidden_dim,recurrent_regularizer=regularizers.l2(1e-8),return_sequences=True))(x)
    #x_gru_2= Bidirectional(CuDNNGRU(hidden_dim,recurrent_regularizer=regularizers.l2(1e-7),return_sequences=True))(x_gru_1)
    #x_gru_1= Dropout(0.1)(x_gru_1)recurrent_regularizer=regularizers.l2(1e-8)
    #x_gru_11= Dropout(0.1)(x_gru_1)
    #x_gru_2 = Bidirectional(CuDNNGRU(hidden_dim,recurrent_regularizer=regularizers.l2(1e-8),return_sequences=True))(x_gru_1)
    #x_com = concatenate([x_gru_2,x_gru_1])
    #x_att_1 = AttentionWeightedAverage()(x_com)
    #x_att_1= Dropout(0.25)(x_att_1)
    #x= Bidirectional(GRU(64,activation='relu',recurrent_dropout=0.1,return_sequences=True))(x)
    #x_att_1= Dropout(0.3)(x_att_1)
    #x_ave=GlobalAveragePooling1D()(x_gru_1)
    #x_ave= Dropout(0.25)(x_ave)
    """
    X_shortcut1 = x

    x_1= Conv1D(filters=250, kernel_size=4, strides=3)(x)
    x_1 = Activation('relu')(x_1)

    x = Conv1D(filters=250, kernel_size=4, strides=3)(x_1)
    x = Activation('relu')(x)

     # connect shortcut to the main path,reshape first
    X_shortcut1=Conv1D(250, kernel_size=1, padding='same', activation='linear')(X_shortcut1)
    X_shortcut1 = Activation('relu')(X_shortcut1)  # pre activation
    x= Add()([X_shortcut1,x])

    x= MaxPool1D(pool_size=3, strides=2, padding='valid')(x)
    """
    
  # block_2
    hidden_dim=50  #250

    convs = []
    filters=[3,5]
    #for ngram in filters:
    ngram=4
    X_shortcut1 = embedded_sequences

    drop_ratio=0.15
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

    #x = Dropout(0.1)(x)
    
    X_shortcut2 = x

    x = Conv1D(filters=hidden_dim,padding='same', kernel_size=ngram)(x)
    x= BatchNormalization()(x)
    x = Dropout(drop_ratio)(x)
    x = PReLU()(x)
    x = Conv1D(filters=hidden_dim,padding='same', kernel_size=ngram)(x)
    x= BatchNormalization()(x)
    x = Dropout(drop_ratio)(x)
    x = PReLU()(x)

  # connect shortcut to the main path
# pre activation
    x = Add()([X_shortcut2,x])

    x = MaxPool1D(pool_size=4,strides=2, padding='valid')(x)
    
    X_shortcut3 = x
    
    x = Conv1D(filters=hidden_dim,padding='same', kernel_size=ngram)(x)
    x= BatchNormalization()(x)
    x = Dropout(drop_ratio)(x)
    x = PReLU()(x)
    x = Conv1D(filters=hidden_dim,padding='same', kernel_size=ngram)(x)
    x= BatchNormalization()(x)
    x = Dropout(drop_ratio)(x)
    x = PReLU()(x)
    
    
    x = Add()([X_shortcut3,x])
    
    x = MaxPool1D(pool_size=4,strides=2, padding='valid')(x)
    
    
    X_shortcut4 = x
    
    x = Conv1D(filters=hidden_dim,padding='same', kernel_size=ngram)(x)
    x= BatchNormalization()(x)
    x = Dropout(drop_ratio)(x)
    x = PReLU()(x)
    x = Conv1D(filters=hidden_dim,padding='same', kernel_size=ngram)(x)
    x= BatchNormalization()(x)
    x = Dropout(drop_ratio)(x)
    x = PReLU()(x)
    
    x = Add()([X_shortcut4,x])
    
    x = MaxPool1D(pool_size=4,strides=2, padding='valid')(x)
        
    X_shortcut5 = x
    
    x = Conv1D(filters=hidden_dim,padding='same', kernel_size=ngram)(x)
    x= BatchNormalization()(x)
    x = Dropout(drop_ratio)(x)
    x = PReLU()(x)
    x = Conv1D(filters=hidden_dim,padding='same', kernel_size=ngram)(x)
    x= BatchNormalization()(x)
    x = Dropout(drop_ratio)(x)
    x = PReLU()(x) 
    
    x = Add()([X_shortcut5,x])
    
    x = MaxPool1D(pool_size=4,strides=2, padding='valid')(x)
    
    X_shortcut6 = x
    
    x = Conv1D(filters=hidden_dim,padding='same', kernel_size=ngram)(x)
    x= BatchNormalization()(x)
    x = Dropout(drop_ratio)(x)
    x = PReLU()(x)
    x = Conv1D(filters=hidden_dim,padding='same', kernel_size=ngram)(x)
    x= BatchNormalization()(x)
    x = Dropout(drop_ratio)(x)
    x = PReLU()(x) 
    
    x = Add()([X_shortcut6,x])
    
    x = GlobalMaxPool1D()(x)
    
    x = Dense(256, activation='linear')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)
    
    #x = Flatten()(x)
    #x = Dropout(0.5)(x)
    #convs.append(x)
    
    #x= Merge(mode='concat', concat_axis=1)(convs)
    #x_2=x_att_1
    #x = Dense(64, activation="relu")(x)
    #x_ave= Dropout(0.275)(x_ave)
    #x_max=GlobalMaxPool1D()(x_gru_1)
    #x_max= Dropout(0.25)(x_max)
    #x_max= Dropout(0.3)(x_max)
    #x_att_2 =Attention()(x_gru_1)
    #x_att_2= Dropout(0.3)(x_att_2)
    #l_dense_2 = TimeDistributed(Dense(128))(x_lstm_2)
    #activation='elu'
    #l_dense_1 =TimeDistributed(Dense(160))(x_gru_1)
    #x_dense= concatenate([x_max,x_ave])
    
    #x_att_3 =AttentionWithContext()(l_dense_2)
    #x_att_3= Dropout(0.2)(x_att_3)
    #x = concatenate([x_att_1,x_att_3])
    #x_dense=BatchNormalization()(x_dense)
    #x= Dropout(0.35)(x)
    """
    x_2=SpatialDropout1D(0.09)(embedded_sequences)
    x_2=Conv1D(filters=hidden_dim, kernel_size=3, strides=1, padding='same', activation='relu')(x_2)
    x_2 = MaxPool1D(2,padding='same')(x_2)
    x_2 = Bidirectional(GRU(hidden_dim,activation='relu',recurrent_dropout=0.1,return_sequences=True))(x_2)
    x_2 = Dropout(0.11)(x_2)
    x_2=GlobalMaxPool1D()(x_2)
    
    x = concatenate([x_1, x_2])
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    """
    """
    #x_1=SpatialDropout1D(0.12)(x_1)
    #x_1 = Bidirectional(GRU(128,activation='relu',recurrent_dropout=0.15,return_sequences=True))(x_1)
    #x_1 = Dropout(0.2)(x_1)
    #x_3 =Attention()(x)
    x_3 =AttentionWithContext()(x)
    #x_3 = Dropout(0.15)(x_3)
    #x_3= Bidirectional(GRU(hidden_dim,activation='relu',recurrent_dropout=0.1,return_sequences=False))(x_3)
    
    #x_3f =Attention()(x_f)
    #x_3f = Dropout(0.15)(x_3f)   #0.12
    #x_3 = concatenate([x_3,x_3f])
    x_3=Dropout(0.2)(x_3)
    #x_3=BatchNormalization()(x_3)
    
    
    #x = concatenate([x_1,x_2,x_3])
    """
    """
    convs = []
    filter_sizes = [3,4,5]
    n_filters=256
    for ngram in filter_sizes:
       X_shortcut = x
       l_conv = Conv1D(nb_filter=n_filters,kernel_regularizer=regularizers.l2(1e-5),border_mode='valid',filter_length=ngram)(x)
       #l_conv= Activation('relu')(l_conv)
       #l_conv = Conv1D(nb_filter=n_filters,kernel_regularizer=regularizers.l2(1e-5),filter_length=fsz)(x)
       #l_conv= Activation('relu')(l_conv)
       #embedding_reshape=Conv1D(nb_filter=n_filters,filter_length=fsz,activation='linear')(X_shortcut)
       #pre_activation=Activation('relu')(embedding_reshape)
       #l_conv = Add()([pre_activation,l_conv])
       l_pool = MaxPool1D(pool_length=maxlen-ngram+1)(l_conv)
       l_flat = Flatten()(l_pool)
       l_flat= Dropout(0.25)(l_flat)
       convs.append(l_flat)
    
    l_merge = Merge(mode='concat', concat_axis=1)(convs)
    #l_cov1= Conv1D(128, 5,kernel_regularizer=regularizers.l2(1e-8),activation='relu',strides=1)(l_merge)
    #l_cov1= Dropout(0.35)(l_cov1)
    #l_pool1 = MaxPool1D(5)(l_cov1)
    #l_cov2 = Conv1D(64, 5, activation='relu')(l_pool1)
    #l_pool2 = MaxPool1D(pool_size=2)(l_cov1)
    #l_flat =Attention()(l_cov1)
    #l_merge = MaxPool1D(2)(l_cov1)
    #l_flat = Flatten()(l_pool2)
    #l_flat =GlobalMaxPool1D()(l_cov2)
    #l_dense = Dense(256, activation='elu')(l_flat)
    #l_dense = Dropout(0.4)(l_dense)
    l_dense = Dense(256, activation='elu')(l_merge)
    l_dense = Dropout(0.35)(l_dense)
    l_dense = Dense(128, activation='elu')(l_dense)
    l_dense = Dropout(0.25)(l_dense)
    x=l_dense
    #preds = Dense(2, activation='softmax')(l_dense)
    """
    """
    embedded_sequences=SpatialDropout1D(0.1)(embedded_sequences)
    embedded_sequences_2=SpatialDropout1D(0.1)(embedded_sequences_2)
    embedded_sequences = concatenate([embedded_sequences,embedded_sequences_2])
    embedded_sequences_1=SpatialDropout1D(0.1)(embedded_sequences)
    conv1 = Conv1D(filters=100, kernel_size=3, activation='relu')(embedded_sequences_1)
    drop1 = Dropout(0.35)(conv1)
    pool1 = MaxPool1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # channel 2
    #inputs2 = Input(shape=(length,))
    #embedding2 = Embedding(vocab_size, 100)(inputs2)
    embedded_sequences_2=SpatialDropout1D(0.1)(embedded_sequences)
    conv2 = Conv1D(filters=100, kernel_size=4, activation='relu')(embedded_sequences_2)
    drop2 = Dropout(0.35)(conv2)
    pool2 = MaxPool1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    # channel 3
    #inputs3 = Input(shape=(length,))
    #embedding3 = Embedding(vocab_size, 100)(inputs3)
    embedded_sequences_3=SpatialDropout1D(0.1)(embedded_sequences)
    conv3 = Conv1D(filters=100, kernel_size=5, activation='relu')(embedded_sequences_3)
    drop3 = Dropout(0.35)(conv3)
    pool3 = MaxPool1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
    # merge
    merged = concatenate([flat1, flat2, flat3])
    # interpretation
    merged= Dropout(0.35)(merged)
    dense1 = Dense(150, activation='elu')(merged)
    #outputs = Dense(1, activation='sigmoid')(dense1)
    x=dense1
    """
    
    """ 
    
    #x_1=SpatialDropout1D(0.01)(embedded_sequences)
    #x_lstm = Bidirectional(LSTM(128,activation='tanh',recurrent_dropout=0.3,return_sequences=True))(x_1)
    #x_1 = Dropout(0.1)(x_1)
    #x_1 = Bidirectional(LSTM(64,activation='tanh',recurrent_dropout=0.15,return_sequences=True))(x_1)
    #x_1 = Bidirectional(GRU(64,activation='tanh',recurrent_dropout=0.1,return_sequences=True))(x_1)
    #x = concatenate([x_lstm,x_1])
    #x_att = AttentionWeightedAverage()(x)
    #x_1 = Bidirectional(GRU(128,activation='relu',recurrent_dropout=0.25,return_sequences=True))(x_1)
    #x=Dropout(0.3)(x_att)
    #x_1 = Bidirectional(LSTM(128,activation='relu',recurrent_dropout=0.3,return_sequences=True))(x_1)
    #x_1 = Dropout(0.25)(x_1)
    #x_1 = Bidirectional(GRU(64,activation='relu',recurrent_dropout=0.25,return_sequences=True))(x_1)
    #x_1 = Dropout(0.2)(x_1)
    #x_1=SpatialDropout1D(0.15)(x_1)
    
    #x_1=Seq2Seq(input_dim=hidden_dim, input_length=embed_size,
               #hidden_dim=128, output_length=maxlen, output_dim=64, depth=4,peek=True)(x_1)
    #x_1 = Bidirectional(GRU(64,activation='relu',recurrent_dropout=0.25,return_sequences=False))(x_1)
    #x_1 = Dropout(0.22)(x_1)
    #x_1=AttentionSeq2Seq(input_dim=embed_size, input_length=embed_size,
                         #hidden_dim=128, output_length=maxlen, output_dim=64, depth=2)(x_1)
    """
    """
    x_1 = GlobalMaxPool1D()(x_1)
    x_1 = Dense(64, activation="relu")(x_1)
    x_1 = Dropout(0.1)(x_1)
    x=x_1
    """
    
    
    """
    auxiliary_input = Input(shape=(11,), name='aux_input')
    x_2 = Dense(64, activation="tanh")(auxiliary_input)
    x_2 = Dropout(0.3)(x_2)
    #x_2=BatchNormalization()(x_2)
    x_2 = Dense(32, activation="relu")(x_2)
    x_2 = Dropout(0.15)(x_2)
    """
    #==================
    
    
    
    #embedded_sequences=SpatialDropout1D(0.02)(embedded_sequences)
    #hidden_dim=200
    
    #embedded_sequences=SpatialDropout1D(0.2)(embedded_sequences)
    #l_gru= Bidirectional(CuDNNGRU(hidden_dim,recurrent_regularizer=regularizers.l2(1e-8),return_sequences=True))(embedded_sequences)
    #l_lstm= Dropout(0.12)(l_lstm)
    #l_gru = TimeDistributed(Dense(200))(l_gru)
    #l_dense=BatchNormalization()(l_dense)
    #x_max=GlobalMaxPool1D()(l_gru)
    #x_ave=GlobalAveragePooling1D()(l_gru)
    #l_gru=Conv1D(200, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(l_gru)
    #l_att = AttentionWithContext()(l_gru)
    #x_encode= concatenate([l_att,x_max,x_ave])
    #l_att= Dropout(0.1)(l_att) inputs=[main_input,Ngram_input], outputs=x
    #sentEncoder = Model(inputs=main_input,outputs=l_att)
   
    
    
    
    #MAX_SENTS=200
    
    #review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), name='main_input', dtype='int32')
    #review_encoder = TimeDistributed(sentEncoder)(review_input)   #0.2
    #l_gru_sent = Bidirectional(CuDNNGRU(hidden_dim,recurrent_regularizer=regularizers.l2(1e-8),return_sequences=True))(review_encoder)
    #x_com = concatenate([review_encoder,l_lstm_sent])
    #x_att_1 = AttentionWeightedAverage()(l_lstm_sent)
    #x_att_1= Dropout(0.4)(x_att_1)
    #l_ave_sent = GlobalAveragePooling1D()(l_lstm_sent)
    #l_ave_sent = Dropout(0.255)(l_ave_sent)
    #l_dense_sent = TimeDistributed(Dense(200))(l_gru_sent)
    #l_gru_sent=Conv1D(200, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(l_gru_sent)
    #l_att_sent = AttentionWithContext()(l_gru_sent)
    #x_ave_sent=GlobalAveragePooling1D()(l_gru_sent)
    #l_att_sent = Dropout(0.4)(l_att_sent)
    #l_att_sent=BatchNormalization()(l_att_sent)
    #l_att_sent = Dropout(0.4)(l_att_sent)
    #l_max_sent = GlobalMaxPool1D()(l_lstm_sent)
    #l_max_sent = Dropout(0.255)(l_max_sent)
    #x_1= concatenate([l_att_sent,x_max_sent,x_ave_sent])
    #x_1=l_att_sent
    
    #x_1=l_att_sent
    #x_1= Dropout(0.25)(x_1)
    
    ########
    """
    hidden_dim=80
    embedded_sequences_2=SpatialDropout1D(0.2)(embedded_sequences_2)
    l_gru_2 = Bidirectional(CuDNNGRU(hidden_dim,recurrent_regularizer=regularizers.l2(1e-8),return_sequences=True))(embedded_sequences_2)
    #l_lstm= Dropout(0.12)(l_lstm)
    #l_dense_2 = TimeDistributed(Dense(200))(l_lstm_2)
    #l_dense=BatchNormalization()(l_dense)
    x_max_2=GlobalMaxPool1D()(l_gru_2)
    x_ave_2=GlobalAveragePooling1D()(l_gru_2)
    l_att_2 = AttentionWithContext()(l_gru_2)
    x_encode_2= concatenate([l_att_2,x_max_2,x_ave_2])
    #l_att= Dropout(0.1)(l_att)
    sentEncoder_2 = Model(inputs=Ngram_input,outputs=x_encode_2)
    
    review_input_2 = Input(shape=(MAX_SENTS_2,MAX_SENT_LENGTH_2),name='aux_input',dtype='int32')
    review_encoder_2 = TimeDistributed(sentEncoder_2)(review_input_2)   #0.2
    l_gru_sent_2 = Bidirectional(CuDNNGRU(hidden_dim,recurrent_regularizer=regularizers.l2(1e-8),return_sequences=True))(review_encoder_2)
    #x_com = concatenate([review_encoder,l_lstm_sent])
    #x_att_1 = AttentionWeightedAverage()(l_lstm_sent)
    #x_att_1= Dropout(0.4)(x_att_1)
    #l_ave_sent = GlobalAveragePooling1D()(l_lstm_sent)
    #l_ave_sent = Dropout(0.255)(l_ave_sent)
    #l_dense_sent_2 = TimeDistributed(Dense(200))(l_lstm_sent_2)
    l_att_sent_2 = AttentionWithContext()(l_gru_sent_2)
    x_max_sent_2=GlobalMaxPool1D()(l_gru_sent_2)
    x_ave_sent_2=GlobalAveragePooling1D()(l_gru_sent_2)
    #l_att_sent = Dropout(0.4)(l_att_sent)
    #l_att_sent=BatchNormalization()(l_att_sent)
    #l_att_sent_2 = Dropout(0.4)(l_att_sent_2)
    #l_max_sent = GlobalMaxPool1D()(l_lstm_sent)
    #l_max_sent = Dropout(0.255)(l_max_sent)
    x_2= concatenate([l_att_sent_2,x_max_sent_2,x_ave_sent_2])
    #x_2=l_att_sent_2
    x_2= Dropout(0.1)(x_2)
    """
    
   
    """
    l_lstm_sent_2 = Bidirectional(GRU(128,activation='relu',recurrent_dropout=0.2,return_sequences=True))(embedded_sequences)
    l_ave_sent = GlobalAveragePooling1D()(l_lstm_sent_2)
    l_ave_sent = Dropout(0.275)(l_ave_sent)
    l_max_sent = GlobalMaxPool1D()(l_lstm_sent_2)
    l_max_sent = Dropout(0.275)(l_max_sent)
    x = concatenate([l_att_sent,l_ave_sent,l_max_sent])
    """
    #x_1= Dense(2, activation="softmax",kernel_regularizer=regularizers.l2(1e-8))(x)
    #x_2= Dense(2, activation="softmax",kernel_regularizer=regularizers.l2(1e-8))(x)
    #x_3= Dense(2, activation="softmax",kernel_regularizer=regularizers.l2(1e-8))(x)
    #x_4= Dense(2, activation="softmax",kernel_regularizer=regularizers.l2(1e-8))(x)
    #x_5= Dense(2, activation="softmax",kernel_regularizer=regularizers.l2(1e-8))(x)
    #x_6= Dense(2, activation="softmax",kernel_regularizer=regularizers.l2(1e-8))(x)
    #x = concatenate([x_1,x_2,x_3,x_4,x_5,x_6])
  
    #preds = Dense(2, activation='softmax')(l_att_sent)
    
    #x = concatenate([x_1,x_2])
    #x = Dropout(0.1)(x)
    #x = Bidirectional(LSTM(32,recurrent_dropout=0.3, return_sequences=True))(x)
    #x_dense = Dense(128, activation="elu")(x_dense)
    #x = GlobalMaxPool1D()(x)
    #x_dense = Dropout(0.3)(x_dense)
    #x=BatchNormalization()(x)
    #x = GRU(64, recurrent_dropout=0.1,return_sequences=False,activation="relu")(x)
    
    #x_dense=BatchNormalization()(x_dense)
    #x_dense = Dense(256, activation="elu")(x_dense)
    #x_dense= Dropout(0.3)(x_dense)
    #x_dense=BatchNormalization()(x_dense)
    #x_dense = Dense(128, activation="elu")(x_dense)
    #x= LeakyReLU(alpha=.001)(x)
    #x= Dropout(0.45)(x)
    #x=BatchNormalization()(x)
    #x_dense = Dense(84, activation="elu")(x_dense)
    #x= LeakyReLU(alpha=.001)(x)
    #x = Dropout(0.15)(x)
    #x=BatchNormalization()(x)
    ##x=BatchNormalization()(x)
    #x = concatenate([x_dense,x_att_2])
    #x=BatchNormalization()(x)
    x= Dense(6, activation="sigmoid",kernel_regularizer=regularizers.l2(1e-8))(x)
    #model = Model(inputs=[main_input,Ngram_input], outputs=x)
    
    
    
    model = Model(inputs=main_input, outputs=x)
    #model = Model(inputs=[review_input,Ngram_input],outputs=x)
    #model = Model(inputs=, outputs=outputs)
    #sgd = SGD(lr=0.09, decay=1e-5, momentum=0.9, nesterov=True)                  #0.004
    #nadam=Nadam(lr=0.0016, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.0035)
    #adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000015,clipvalue=0.0012)
    
    nadam=Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.0022)
    model.compile(loss='binary_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy',f1_score,auc])
    print(model.summary())
    return model

#nadam=Nadam(lr=0.00135, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.001)
#adam=Adam(lr=0.0015, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000015,clipvalue=0.0012)
batch_size = 400
epochs = 20

#5 fold:
accumulator=[]

from sklearn.metrics import roc_auc_score
K_fold=splits
#pred_val_accumulator=test_accumulator=None


pred_val_accumulator=pickle.load(open("prediction_CNN_fold_"+str(3)+".pkl", "rb")) 
test_accumulator=pickle.load(open("test_average_CNN_fold_"+str(3)+".pkl", "rb"))


count=0
for i in range(4,10):
    print("======")
    print(str(i)+"fold")
    print("======")
    c_train_X = X_tr[train.id.isin(train_ids[i])]
    c_train_y = y_tr[train.id.isin(train_ids[i])]
    c_val_X = X_tr[train.id.isin(val_ids[i])]
    c_val_y = y_tr[train.id.isin(val_ids[i])]       
        #c_train_X_twitter=np.vstack((c_train_XOne_twitter,c_train_XTwo_twitter))
        
    file_path="weights_base_5_fold_CNN.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min",verbose=1, patience=4)
    callbacks_list = [early,checkpoint]
    model = get_model()
    seed(1)
    history=model.fit(c_train_X,c_train_y, batch_size=batch_size, epochs=epochs, 
                      validation_data=(c_val_X, c_val_y), callbacks=callbacks_list) #[c_train_X, c_train_X_gram]
    #history=model.fit({'main_input':c_train_X, 'aux_input': c_train_X_twitter},c_train_y, batch_size=batch_size, epochs=epochs, 
                      #validation_data=({'main_input':c_val_X, 'aux_input': c_val_X_twitter}, c_val_y),callbacks=callbacks_list)
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


  
     ##code validation for NN model
    print(history.history.keys())
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.savefig('k-fold-plot1'+str(i)+'.png', format='png')

 
    print(history.history.keys())
    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.savefig('k-fold-plot2'+str(i)+'.png', format='png')
    
    model.load_weights(file_path)
    pred_val = model.predict(c_val_X,batch_size=batch_size, verbose=1)
    #pred = model.predict( {'main_input':c_val_X, 'aux_input': c_val_X_twitter},batch_size=batch_size, verbose=1)
    y_test = model.predict( X_te_1,batch_size=batch_size, verbose=1)
   
    if(i==0):
         pred_val_accumulator=pred_val
         test_accumulator=y_test
    else:
         #pred_accumulator is not None:
         pred_val_accumulator=np.vstack((pred_val_accumulator,pred_val))
         #test_accumulator is not None:
         test_accumulator=test_accumulator+y_test
    sub_accumulator=[]
    for j in range(0,len(list_classes)):
        result=pred_val[:,j].reshape(-1, 1)
        roc_score=roc_auc_score(c_val_y[:,j].reshape(-1, 1),result)
        print("#Column: "+str(j)+" Roc_auc_score: "+str(roc_score))
        sub_accumulator.append(roc_score)
    print("#Average Roc_auc_score is: {}\n".format( np.mean(sub_accumulator) ))
    pickle.dump(pred_val_accumulator,open("prediction_CNN_fold_"+str(i)+".pkl", "wb"))
    pickle.dump(test_accumulator,open("test_average_CNN_fold_"+str(i)+".pkl", "wb"))
    accumulator.append(np.mean(sub_accumulator))
    del model
test_average=test_accumulator/K_fold

print("#Total average Roc_auc_score is: {}\n".format( np.mean(accumulator) ))    


pickle.dump(pred_val_accumulator,open("prediction_CNN.pkl", "wb"))
if test_accumulator is not None:
   pickle.dump(test_average,open("test_average_CNN.pkl", "wb"))


print("dump test average")

#test_average=pickle.load(open("test_average.pkl", "rb")) 



seed(1)
model = get_model()
batch_size =360
epochs = 50
print("start training")
file_path="weights_base_256_CNN.hdf5"
#file_path="weights_base_256_LSTM.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min",verbose=1, patience=5)

#model.load_weights(file_path)

callbacks_list = [checkpoint, early] #early
#history=model.fit({'main_input': X_tr_1, 'aux_input': X_tr_2},y_tr, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)


history=model.fit( X_tr_1,y_tr, batch_size=batch_size, epochs=epochs, validation_split=0.09, callbacks=callbacks_list)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



##code validation for NN model
print(history.history.keys())
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.savefig('plot1.png', format='png')

 
print(history.history.keys())
plt.clf()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.savefig('plot2.png', format='png')





print(history.history.keys())
plt.clf()
plt.plot(history.history['f1_score'])
plt.plot(history.history['val_f1_score'])
plt.title('model f1_score')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.savefig('plot3.png', format='png')


print(history.history.keys())
plt.clf()
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('model auc')
plt.ylabel('auc score')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.savefig('plot4.png', format='png')







model.load_weights(file_path)

print("start testing")
#y_test = model.predict( {'main_input':X_te_1, 'aux_input': X_te_2},batch_size=batch_size, verbose=1)

batch_size =1024

y_test = model.predict( X_te_1,batch_size=batch_size, verbose=1)



sample_submission = pd.read_csv("sample_submission.csv")

sample_submission[list_classes] = y_test



sample_submission.to_csv("baseline_DPCNN.csv", index=False)