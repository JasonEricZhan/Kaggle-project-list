from keras.models import  Sequential,Model
from keras.layers.core import   Dense
from keras.layers  import LSTM,GRU,Bidirectional,Dropout
from keras.callbacks import EarlyStopping


from keras import regularizers
from keras import initializers
from keras.optimizers import Adam

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error

from numpy.random import seed
seed(1)

%matplotlib inline
import matplotlib.pyplot as plt;

count_prev=0 
mae_list=[]
RMSL_list=[]
for i in np.arange(0,10,2):
  length_first_month=len(df.loc[(df['month']==i)]) #first month for training
  length_second_month=len(df.loc[(df['month']==i+2)])#second month for training
  length_val=len(df.loc[df['month']==i+4])#Third month for validation
  length_train=length_first_month+length_second_month #length for training
  count_now=count_prev+length_train              
  train=df_norm[count_prev:count_now]
  val=df_norm[count_now:count_now+length_val]
  
  train=train.reshape(-1,1,17)      #it depends on the number of features
  val=val.reshape(-1,1,17)
  
  y_train=df.POLYLINE_time_second_log[count_prev:count_now]
  y_val=df.POLYLINE_time_second_log[count_now:count_now+length_val]

  train_answer=df.POLYLINE_time_second[count_prev:count_now]
  test_answer=df.POLYLINE_time_second[count_now:count_now+length_val]
  

  count_prev=count_prev+length_first_month     #move one month 

  model = Sequential()
  model.add(Bidirectional(LSTM(128,recurrent_dropout=0.3,
                              return_sequences = True),input_shape=(1,train.shape[2])))
  model.add(Dropout(0.2))
  model.add(Bidirectional(LSTM(128,recurrent_dropout=0.3,
                              return_sequences = True)))
  model.add(Dropout(0.2))
  model.add(Bidirectional(LSTM(64,recurrent_dropout=0.3,
                              return_sequences = True)))
  model.add(Dropout(0.2))
  model.add(Bidirectional(LSTM(64,recurrent_dropout=0.3,
                              return_sequences = True)))
  model.add(Dropout(0.2))
  
  model.add(Bidirectional(LSTM(32,recurrent_dropout=0.3,
                              return_sequences = False)))
  model.add(Dropout(0.2))
  model.add(Dense(1))
  model.add(Activation('linear'))
  print(model.summary())
 
  
  adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,clipvalue=0.0015)
  model.compile(loss='mean_squared_logarithmic_error', optimizer=adam,metrics =['mae'])
  

  earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0)
  history = model.fit(train,y_train.values,batch_size=64,validation_data=(val, y_val),
                      epochs=40,shuffle=False,callbacks=[earlystop])
  pred=model.predict(val)
  
  ##code validation for NN model
  print(history.history.keys())
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model accuracy')
  plt.ylabel('mean squared logarithmic error')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()
    
  print(history.history.keys())
  plt.plot(history.history['mean_absolute_error'])
  plt.plot(history.history['val_mean_absolute_error'])
  plt.title('model accuracy')
  plt.ylabel('mean absolute error')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()
  

 
  pred=pred.reshape(-1,)

  mae=mean_absolute_error(np.exp(pred),test_answer)
  msle=mean_squared_log_error(np.exp(pred),test_answer)
  
  
  mae_list.append(mae)
  RMSL_list.append(np.sqrt(msle))
  
  
  print ("# MAE: {}\n".format( mae )	)
  print ("# Mean square log error: {}\n".format( msle )	)
  print ("# Root Mean square log error: {}\n".format( np.sqrt(msle ))	)

print ("# mean of MAE: {}\n".format( np.mean(mae_list) )	)
print ("# std of MAE: {}\n".format( np.std(mae_list) )	)
print ("# mean of Root Mean square log error: {}\n".format( np.mean(RMSL_list ))	)
print ("# std of Root Mean square log error: {}\n".format( np.std(RMSL_list) )	)