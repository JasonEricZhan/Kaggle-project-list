
#this script is to hash the out of fold back to orignial data's order

import pickle
import pandas as pd
import numpy as np



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')



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



oof_Capsule=pickle.load(open("prediction_RNN_Capsule.pkl", "rb"))
oof_CNN=pickle.load(open("prediction_DPCNN.pkl", "rb"))
oof_GRU=pickle.load(open("prediction_GRU.pkl", "rb"))
oof_GRU_LSTM=pickle.load(open("test_average_LSTM.pkl", "rb"))


#=============================Example===================================

container_before=oof_CNN  #change to other fold file
length_before=0
length_after=0
new=np.zeros((container_before.shape[0],container_before.shape[1]))
for i in range(0,10):   #take the fold
    length_after+=len(np.where(train.id.isin(train_ids[i]).values==0)[0])
    if i ==9:
       input_segment=container_before[length_before:]
    else:
       input_segment=container_before[length_before:length_after]
    length_before+=len(np.where(train.id.isin(train_ids[i]).values==0)[0])
    counter=0
    for j in  np.where(train.id.isin(train_ids[i]).values==0)[0]:   #transfer back to original index
        new[j,:]=input_segment[counter,:]
        counter+=1
     
oof_CNN=new
