from sklearn.metrics import roc_auc_score
K_fold=10
pred_val_accumulator=test_accumulator=None
accumulator=[]
for i in range(1,int(K_fold+1)):
    print("=================================")
    print("Start on: "+str(i)+" fold")
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
        c_train_X=X_tr[:indexOne]
        c_train_y=y_tr[:indexOne]
    else:
        c_train_XOne=X_tr[:indexOne]
        c_train_yOne=y_tr[:indexOne]
        c_train_XTwo=X_tr[indexTwo:]
        c_train_yTwo=y_tr[indexTwo:]
        
        c_train_X=np.vstack((c_train_XOne,c_train_XTwo))
        c_train_y=np.vstack((c_train_yOne,c_train_yTwo))
        
        
        
    print("")
    file_path="weights_base_5_fold_CNN.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min",verbose=1, patience=5)
    callbacks_list = [early,checkpoint]
    model = get_model()
    seed(42)   #original is 1, change to 42 
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
   
    if(i==1):
         pred_val_accumulator=pred_val
         test_accumulator=y_test
    else:
         pred_val_accumulator=np.vstack((pred_val_accumulator,pred_val))
         test_accumulator=test_accumulator+y_test
    sub_accumulator=[]
    for j in range(0,len(list_classes)):
        result=pred_val[:,j].reshape(-1, 1)
        roc_score=roc_auc_score(c_val_y[:,j].reshape(-1, 1),result)
        print("#Column: "+str(j)+" Roc_auc_score: "+str(roc_score))
        sub_accumulator.append(roc_score)
    print("#Average Roc_auc_score is: {}\n".format( np.mean(sub_accumulator) ))
    #prevent the program break down(core dump), save to the disk at each step
    pickle.dump(pred_val_accumulator,open("prediction_RNN_1_"+str(i)+".pkl", "wb"))
    pickle.dump(test_accumulator,open("test_average_1_"+str(i)+".pkl", "wb"))
    accumulator.append(np.mean(sub_accumulator))
    del model
test_average=test_accumulator/K_fold

print("#Total average Roc_auc_score is: {}\n".format( np.mean(accumulator) ))    


pickle.dump(pred_val_accumulator,open("prediction_RNN.pkl", "wb"))
if test_accumulator is not None:
   pickle.dump(test_average,open("test_average.pkl", "wb"))
