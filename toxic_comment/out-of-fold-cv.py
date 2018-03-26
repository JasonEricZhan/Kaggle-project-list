from sklearn.metrics import roc_auc_score
import numpy as np

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



from sklearn.metrics import roc_auc_score
splits

accumulator=[]
for i in range(splits):
    print("======")
    print(str(i)+"fold")
    print("======")
    c_train_X = X_tr[train.id.isin(train_ids[i])]
    c_train_y = y_tr[train.id.isin(train_ids[i])]
    c_val_X = X_tr[train.id.isin(val_ids[i])]
    c_val_y = y_tr[train.id.isin(val_ids[i])]       
        #c_train_X_twitter=np.vstack((c_train_XOne_twitter,c_train_XTwo_twitter))
        
    """"
    
    
    NN model training
    
    """
    #record the curve plot
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
    pickle.dump(pred_val_accumulator,open("OOF_"+str(i)+".pkl", "wb"))
    pickle.dump(test_accumulator,open("test_average_"+str(i)+".pkl", "wb"))
    accumulator.append(np.mean(sub_accumulator))
    del model
test_average=test_accumulator/K_fold

print("#Total average Roc_auc_score is: {}\n".format( np.mean(accumulator) ))    


pickle.dump(pred_val_accumulator,open("OOF_.pkl", "wb"))
if test_accumulator is not None:
   pickle.dump(test_average,open("test_average_.pkl", "wb"))

