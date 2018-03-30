

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
import xgboost as xgb




train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y= train[list_classes].values      


import pickle




import pickle

test_ave_CNN=pickle.load(open("stack_pack/test_average_CNN.pkl", "rb"))
test_ave_Capsule=pickle.load(open("stack_pack/test_average_Capsule.pkl", "rb"))
test_ave_GRU=pickle.load(open("stack_pack/test_average_0.9854.pkl", "rb"))
test_ave_LSTM=pickle.load(open("stack_pack/test_average_LSTM_0.9852.pkl", "rb"))
#test_ave_GRU_2=pickle.load(open("test_average.pkl", "rb"))
test_ave_GRU_3=pickle.load(open("test_ave_wiki_char_gru.pkl", "rb"))
test_ave_lgbm=pd.read_csv('lvl0_lgbm_clean_sub.csv').iloc[:,1:].values
test_ave_no_char=pickle.load(open("test_average__GRU_no_char9.pkl", "rb"))/10


oof_Capsule=pickle.load(open("stack_pack/prediction_RNN_Capsule.pkl", "rb"))
oof_CNN=pickle.load(open("stack_pack/prediction_DPCNN.pkl", "rb"))
oof_GRU=pickle.load(open("stack_pack/prediction_RNN _0.9854.pkl", "rb"))
#oof_GRU_2=pickle.load(open("prediction_RNN.pkl", "rb"))
oof_GRU_3=pickle.load(open("wiki_char_oof.pkl", "rb"))
oof_LSTM=pickle.load(open("stack_pack/prediction_RNN_LSTM_0.9852.pkl", "rb"))
oof_GRU_no_char=pickle.load(open("oof_GRU_no_char.pkl", "rb"))
oof_lgbm=pd.read_csv('lvl0_lgbm_clean_oof.csv').iloc[:,7:].values

class_names = list(train)[-6:]
multarray = np.array([100000, 10000, 1000, 100, 10, 1])
y_multi = np.sum(train[class_names].values * multarray, axis=1)



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
    
    

container_before=oof_CNN
length_before=0
length_after=0
new=np.zeros((container_before.shape[0],container_before.shape[1]))
for i in range(0,10):
    length_after+=len(np.where(train.id.isin(train_ids[i]).values==0)[0])
    if i ==9:
       input_segment=container_before[length_before:]
    else:
       input_segment=container_before[length_before:length_after]
    length_before+=len(np.where(train.id.isin(train_ids[i]).values==0)[0])
    counter=0
    for j in  np.where(train.id.isin(train_ids[i]).values==0)[0]:
        new[j,:]=input_segment[counter,:]
        counter+=1
    #print(counter)
     
oof_CNN=new





container_before=oof_Capsule
length_before=0
length_after=0
new=np.zeros((container_before.shape[0],container_before.shape[1]))
for i in range(0,10):
    length_after+=len(np.where(train.id.isin(train_ids[i]).values==0)[0])
    if i ==9:
       input_segment=container_before[length_before:]
    else:
       input_segment=container_before[length_before:length_after]
    length_before+=len(np.where(train.id.isin(train_ids[i]).values==0)[0])
    counter=0
    for j in  np.where(train.id.isin(train_ids[i]).values==0)[0]:
        new[j,:]=input_segment[counter,:]
        counter+=1
    #print(counter)
     
oof_Capsule=new








"""
container_before=oof_GRU_3
length_before=0
length_after=0
new=np.zeros((container_before.shape[0],container_before.shape[1]))
for i in range(0,10):
    length_after+=len(np.where(train.id.isin(train_ids[i]).values==0)[0])
    if i ==9:
       input_segment=container_before[length_before:]
    else:
       input_segment=container_before[length_before:length_after]
    length_before+=len(np.where(train.id.isin(train_ids[i]).values==0)[0])
    counter=0
    for j in  np.where(train.id.isin(train_ids[i]).values==0)[0]:
        new[j,:]=input_segment[counter,:]
        counter+=1
    #print(counter)
     
oof_GRU_3=new
"""    




container_before=oof_LSTM
length_before=0
length_after=0
new=np.zeros((container_before.shape[0],container_before.shape[1]))
for i in range(0,10):
    length_after+=len(np.where(train.id.isin(train_ids[i]).values==0)[0])
    if i ==9:
       input_segment=container_before[length_before:]
    else:
       input_segment=container_before[length_before:length_after]
    length_before+=len(np.where(train.id.isin(train_ids[i]).values==0)[0])
    counter=0
    for j in  np.where(train.id.isin(train_ids[i]).values==0)[0]:
        new[j,:]=input_segment[counter,:]
        counter+=1
    #print(counter)
     
oof_LSTM=new    
    
    
    
    
    
    
    
    
"""
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

stacker = lgb.LGBMClassifier(max_depth=4, metric="auc", n_estimators=50, num_leaves=15, boosting_type="gbdt", learning_rate=0.01, feature_fraction=0.45, colsample_bytree=0.45, bagging_fraction=0.8, bagging_freq=5, reg_lambda=0.2
                            )


sub = pd.read_csv("sample_submission.csv")







# Fit and submit
scores = []
for label in list_classes:
    print(label)
    score = cross_val_score(stacker, X_tr, train[label], cv=10, scoring='roc_auc')
    print("AUC:", score)
    scores.append(np.mean(score))
    stacker.fit(X_tr, train[label])
    sub[label] = stacker.predict_proba(X_te)[:,1]
print("CV score:", np.mean(scores))

sub.to_csv("ensemble.csv", index=False)

"""


absord=0.5*oof_CNN+oof_Capsule*0.5
absord_te=0.5*test_ave_CNN+test_ave_Capsule*0.5


absord_GRU=oof_GRU_no_char*0.5+oof_GRU_3*0.5
absord_GRU_te=test_ave_GRU_3*0.5+test_ave_no_char*0.5



#absord_GRU=0.5*oof_GRU_2+0.5*oof_GRU
#absord_te_GRU=0.5*test_ave_GRU_2+0.5*test_ave_GRU

print(absord.shape)
print(oof_GRU.shape)
print(oof_LSTM.shape)
print(oof_lgbm.shape)
      
X_tr=np.hstack((absord,oof_GRU,oof_LSTM,oof_lgbm,absord_GRU))      
X_te=np.hstack((absord_te,test_ave_GRU,test_ave_LSTM,test_ave_lgbm,absord_GRU_te))       
      
#oof_GRU    
#test_ave_GRU


gbm = xgb.XGBClassifier(
n_estimators= 1000,
max_depth= 3,
min_child_weight= 2,
gamma=0.9,                        
   #subsample=0.8,
   #colsample_bytree=0.8,
objective= 'multi:softprob',
   #nthread= -1,
nthread=1,
seed=10,
scale_pos_weight=1)


def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=2017, num_rounds=500):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.11 #0.12
    param['max_depth'] = 3  #4
    param['n_estimators']= 1200  #800
    param['silent'] = 1
    #param['max_leaf_nodes'] = 2000
    param['eval_metric'] = 'logloss'
    param['min_child_weight'] = 1
    param['subsample'] = 0.65
    param['colsample_bytree'] = 0.785
    #param['booster']='dart'
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=15)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    if test_y is not None:
       print('ROC AUC:', roc_auc_score( test_y, pred_test_y))
    return model



import xgboost as xgb
from sklearn.metrics import roc_auc_score
K_fold=10
pred_val_accumulator=test_accumulator=None
accumulator=[]
for i in range(splits):
    print("=================================")
    print("Start on: "+str(i)+" fold")
    c_train_X = X_tr[train.id.isin(train_ids[i])]
    c_train_y = y[train.id.isin(train_ids[i])]
    c_val_X = X_tr[train.id.isin(val_ids[i])]
    c_val_y = y[train.id.isin(val_ids[i])]
    

    sub_accumulator=[]
    pred_val=np.zeros((len(c_val_X),len(list_classes)))
    y_test=np.zeros((len(test),len(list_classes)))
    for j in range(0,len(list_classes)):
        #gbm.fit(c_train_X, c_train_y[:,j],eval_set=(c_val_X,c_val_y[:,j]),eval_metric='auc',verbose=True)
        model = runXGB(c_train_X, c_train_y[:,j], c_val_X,c_val_y[:,j])
        pred_val[:,j]=model.predict(xgb.DMatrix(c_val_X))
        y_test[:,j]=model.predict(xgb.DMatrix(X_te))
        #pred_val[:,j]=gbm.predict(c_val_X)
        result=pred_val[:,j].reshape(-1, 1)
        roc_score=roc_auc_score(c_val_y[:,j].reshape(-1, 1),result)
        print("#Column: "+str(j)+" Roc_auc_score: "+str(roc_score))
        sub_accumulator.append(roc_score)
        
    if(i==0):
        pred_val_accumulator=pred_val
        test_accumulator=y_test
    else:
        pred_val_accumulator=np.vstack((pred_val_accumulator,pred_val))
        test_accumulator=test_accumulator+y_test
         
    print("#Average Roc_auc_score is: {}\n".format( np.mean(sub_accumulator) ))
    pickle.dump(pred_val_accumulator,open("second_layer_tr"+str(i)+".pkl", "wb"))
    pickle.dump(test_accumulator,open("second_layer_te"+str(i)+".pkl", "wb"))
    accumulator.append(np.mean(sub_accumulator))
    del model

print("#Total average Roc_auc_score is: {}\n".format( np.mean(accumulator) ))    
print("#std Roc_auc_score is: {}\n".format( np.std(accumulator) ))    

test_accumulator=test_accumulator/10

pickle.dump(pred_val_accumulator,open("second_layer_oof.pkl", "wb"))
if test_accumulator is not None:
   pickle.dump(test_accumulator,open("second_layer.pkl", "wb"))





