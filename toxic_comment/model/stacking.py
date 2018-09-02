

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
#=======================

#........load out of fold.......

#=======================
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
    
    
    
    
    
#=======================

#........hash back.......

#=======================


#absord high correlated data


absord=0.5*oof_CNN+oof_Capsule*0.5
absord_te=0.5*test_ave_CNN+test_ave_Capsule*0.5


absord_GRU=oof_GRU_no_char*0.5+oof_GRU_3*0.5
absord_GRU_te=test_ave_GRU_3*0.5+test_ave_no_char*0.5



print(absord.shape)
print(oof_GRU.shape)
print(oof_LSTM.shape)
print(oof_lgbm.shape)
      
X_tr=np.hstack((absord,oof_GRU,oof_LSTM,oof_lgbm,absord_GRU))      
X_te=np.hstack((absord_te,test_ave_GRU,test_ave_LSTM,test_ave_lgbm,absord_GRU_te))       
      




def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=2017, num_rounds=500):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.11 #0.12
    param['max_depth'] = 3  #4
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
        model = runXGB(c_train_X, c_train_y[:,j], c_val_X,c_val_y[:,j])
        pred_val[:,j]=model.predict(xgb.DMatrix(c_val_X))
        y_test[:,j]=model.predict(xgb.DMatrix(X_te))
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





