
import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,Activation,GRU,Conv1D,SpatialDropout1D,MaxPool1D
from keras.layers import Bidirectional, GlobalMaxPool1D,BatchNormalization,concatenate,TimeDistributed,Merge,Flatten
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.optimizers import Adam,SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Layer  
from keras import initializers, regularizers, constraints  
from keras import backend as K
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')

embed_size = 300 # how big is each word vector
max_features = 80000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 300 # max number of words in a comment to use


#EMBEDDING_FILE="glove.6B.300d.txt"

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


#train = train.sample(frac=1)




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




corpus=df.comment_text


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

print("....start....cleaning")

from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']


from nltk.tokenize import TweetTokenizer


tokenizer=TweetTokenizer()


re_tok = re.compile(r'([1234567890!@#$%^&*_+-=,./<>?;:"[][}]"\'\\|�鎿�𤲞阬威鄞捍朝溘甄蝓壇螞¯岑�''\t])')



def clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    #Convert to lower case , so that Hi and hi are the same
    comment=comment.lower()
    #remove \n
    comment=re.sub("\\n"," ",comment)
    comment=re.sub(r"fucksex","fuck sex",comment)
    comment=re.sub(r"f u c k","fuck",comment)
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
    comment=re.sub(r"mothjer","mother",comment)
    s=comment
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')
    comment=s
    comment = re_tok.sub(' ', comment)
    token='1234567890!@#$%^&*_+-=,./<>?;:"[][}]"\'\\|'

    comment=comment.replace(token," ")
    #sub is replace in python
    
    
    #Split the sentences into words
    words=tokenizer.tokenize(comment)
    
    #print(words)
    # (')aphostophe  replacement (ie)   you're --> you are  
    # ( basic dictionary lookup : master dictionary present in a hidden block of code)
    words=[APPO[word] if word in APPO else word for word in words]
    #print(words)
    words=[lem.lemmatize(word, "v") for word in words]
    #print(words)
    words = [w for w in words if not w in stop_words]
    #print(words)
    #
    
    
    clean_sent=" ".join(words)
    # remove any non alphanum,digit character
    #clean_sent=re.sub("\W+"," ",clean_sent)
    #clean_sent=re.sub("  "," ",clean_sent)
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
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
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
    #data=data.apply(lambda x:lemmatize_sentence(x))
    return data

def multiply_columns_clean_And_lemmatize_sentence(data):
    data = data.apply(lambda x: clean(x))
    data=data.apply(lambda x:lemmatize_sentence(x))
    return data
    
#corpus= parallelize_dataframe(corpus, multiply_columns_clean)
#corpus= parallelize_dataframe(corpus, multiply_columns_clean_And_lemmatize_sentence)

#clean_corpus=corpus.apply(lambda x :clean(x))
import  pickle 
#pickle.dump(corpus,open("tmp.pkl", "wb"))


corpus=pickle.load(open("tmp.pkl", "rb")) 

clean_corpus = corpus
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


#merge['comment_text']=clean_corpus




df["comment_text"]=clean_corpus

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

with open('dirtyWord_bank.txt',"rt") as f:
     dirtyWBank =f.readlines()
with open('dirtyWord_bank2.txt',"rt") as f:
     dirtyWBank2 =f.readlines()

dirtyWBank = [x.strip() for x in dirtyWBank] 

dirtyWBank2 = [x.strip() for x in dirtyWBank2] 

dirty_set=set(dirtyWBank+dirtyWBank2)



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

train_cl=df[:train.shape[0]]
test_cl=df[train.shape[0]:]

test_cl=test_cl.reset_index(drop=True)

#dtrain, dval = train_test_split(train_cl, random_state=2345, train_size=0.8)

dtrain=train_cl


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
y_tr = dtrain[list_classes].values      
#y_val=dval[list_classes].values      
    


list_sentences_train=dtrain.comment_text
#list_sentences_val=dval.comment_text
list_sentences_test=test_cl.comment_text


"""
#5 fold:
from sklearn.metrics import roc_auc_score
K_fold=5
for i in range(1,int(K_fold+1)):
    print("=================================")
    print("Start on: "+str(i)+" fold")
    model = get_model()
    accumulator=[]
    length=int(len(X_tr)/K_fold)
    if((len(X_tr)/K_fold)==length):
        c_val_X=X_tr[length*(i-1):length*i]
        c_val_y=y_tr[length*(i-1):length*i]
    else:
        c_val_X=X_tr[length*(i-1):length*i]
        c_val_y=y_tr[length*(i-1):length*i]
        if(i==K_fold):
            c_val_X=X_tr[length*(i-1):]
            c_val_y=y_tr[length*(i-1):]
    
    indexOne=length*(i-1)
    indexTwo=length*(i)
    
    if(i==1):
        c_train_X=X_tr[indexTwo:]
        c_train_y=y_tr[indexTwo:]
    elif(i==K_fold):
        c_train_X=X_tr[indexOne:]
        c_train_y=y_tr[indexOne:]
    else:
        c_train_XOne=X_tr[:indexOne]
        c_train_yOne=y_tr[:indexOne]
        c_train_XTwo=X_tr[indexTwo:]
        c_train_yTwo=y_tr[indexTwo:]

        c_train_X=np.vstack((c_train_XOne,c_train_XTwo))
        c_train_y=np.vstack((c_train_yOne,c_train_yTwo))
        
    print("")
    file_path="weights_base_256_2.best.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min",verbose=1, patience=10)
    callbacks_list = [checkpoint, early]
    history=model.fit(c_train_X,c_train_y, batch_size=batch_size, epochs=epochs, 
                      validation_data=(c_val_X, c_val_y), callbacks=callbacks_list)
    
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
    
    #model.load_weights(file_path)
    pred = model.predict(c_val_X,batch_size=batch_size, verbose=1)
    sub_accumulator=[]
    for j in range(0,len(list_classes)):
        result=pred[:,j].reshape(-1, 1)
        roc_score=roc_auc_score(c_val_y,result)
        print("#Column: "+str(j)+" Roc_auc_score: "+str(roc_score))
        sub_accumulator.append(roc_score)
    print("#Average Roc_auc_score is: {}\n".format( np.mean(sub_accumulator) ))
    accumulator=accumulator+sub_accumulator

print("#Total average Roc_auc_score is: {}\n".format( np.mean(accumulator) ))    
"""
 


