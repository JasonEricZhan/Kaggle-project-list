"""
Author:Edited and create by Eric

"""
# coding: utf-8

# In[1]:

import pandas as pd
df=pd.read_csv('train.csv').sort_values(['TIMESTAMP'])




# In[2]:

df.columns


# In[3]:

def getFirst_LONGITUDE(x):
   if(len(x)>0):
      return x[0][0]
   else:
      return "no information"

def getFirst_LATITUDE(x):
   if(len(x)>0):
      return x[0][1]
   else:
      return "no information"
   
def getLast_LONGITUDE(x):
   if(len(x)>0):
      return x[-1][0]
   else:
      return "no information"
def getLast_LATITUDE(x):
  if(len(x)>0):
     return x[-1][1]
  else:
     return "no information"
### Get Haversine distance
def get_dist(lonlat):
  if len(lonlat) >0:
     lon_diff = np.abs(lonlat[0][0]-lonlat[-1][0])*np.pi/360.0
     lat_diff = np.abs(lonlat[0][1]-lonlat[-1][1])*np.pi/360.0
     a = np.sin(lat_diff)**2 + np.cos(lonlat[-1][0]*np.pi/180.0) * np.cos(lonlat[0][0]*np.pi/180.0) * np.sin(lon_diff)**2  
     d = 2*6371*np.arctan2(np.sqrt(a), np.sqrt(1-a))
     return(d)
  else:
     return 0


# In[4]:

preprocessing


# In[5]:

def time_processing(df): 
   print("at time processing....")
   data=df.copy()
   data['TIMESTAMP']=data['TIMESTAMP']-1372636800
   data['day_of_year']=(data['TIMESTAMP']/86400).astype(int)  #day of year
   data['month']=(data['TIMESTAMP']/(86400*30)).astype(int)  #month
   data['day_of_month']=(data['day_of_year']-data['month'].astype(int)*30)
   data['day_of_week']=(data['day_of_month']%7)+1
   data['month']=data['month'].astype(int)
   data['day_of_month']=data['day_of_month'].astype(int)
   data['hour_in_day_not_int']=(data['TIMESTAMP']-data['month']*30*86400-data['day_of_month']*86400)/(60*60)
   data['hour_in_day']=data['hour_in_day_not_int'].astype(int)
   data['minute_in_hour']=(data['hour_in_day_not_int']-data['hour_in_day'])*60
   data['minute_in_hour']=data['minute_in_hour'].astype(int)
   
   return data


# In[8]:

def geography_processing(df): 
    print("at geography processing....")
    data=df.copy()
    data['POLYLINE_time_second']=data['POLYLINE'].apply(lambda x: (len(x)-1)*15)
    #print(data.describe())
    data=data[data['POLYLINE_time_second']<10000] #testing data range
    data=data[data['POLYLINE_time_second']>0]
    #print(data.describe())
    data['start_longitude']=data['POLYLINE'].apply(getFirst_LONGITUDE)
    data['start_latitude']=data['POLYLINE'].apply(getFirst_LATITUDE)
    data['end_longitude']=data['POLYLINE'].apply(getLast_LONGITUDE)
    data['end_latitude']=data['POLYLINE'].apply(getLast_LATITUDE)
    data['dist']=data['POLYLINE'].apply(get_dist)
    
    return data


# In[ ]:

import json
from sklearn.utils import resample
def sampling_fliter_machine(df,numberOfSample=0,proportion=0,double_hour=False,random_state=0):
    """
    double_hour is to emphazie particular hour or not
    """
    accumulator=[]
    data=df.copy()
    data=time_processing(data)
    #print(data.describe())
    data=data.loc[data['hour_in_day']>=2]
    data=data.loc[data['hour_in_day']<18]
    for i in range(0,13):
        data_month=data.loc[data['month']==i].copy()
        #print(data_month.describe())
        data_month['POLYLINE'] = data_month['POLYLINE'].apply(json.loads)
        data_month=geography_processing(data_month)
            #print(data_month.describe())
        data_month=data_month.loc[data_month['start_longitude']<=-7]
        data_month=data_month.loc[data_month['start_longitude']>=-9]
        data_month=data_month.loc[data_month['start_latitude']>=40]
        data_month=data_month.loc[data_month['start_latitude']<=42]
        data_month=data_month.loc[data_month['end_longitude']<=-7]
        data_month=data_month.loc[data_month['end_longitude']>=-9]
        data_month=data_month.loc[data_month['end_latitude']>=40]
        data_month=data_month.loc[data_month['end_latitude']<=42]
        data_month.reset_index( inplace = True, drop = True )
        print("Now is "+str(i)+" month:")
        for j in range(2,18):
            set_=data_month.loc[data_month['hour_in_day']==j]
            #print(set_)
            if(numberOfSample>0):
                if len(set_)==0:
                    print("**Warning: month"+str(i)+" hour "+str(j)+" 's data is empty**")
                    pass
                else:
                    if len(set_)<numberOfSample:
                       print("**Little Warning: month"+str(i)+" hour "+str(j)+" 's data is not enough**")
                       sample=set_
                    else:
                       if(double_hour==True):
                          if((j==3) or (j==8) or (j==14) or (j==17)):
                            if(numberOfSample*2<len(set_)):
                                sample=resample(set_,n_samples=numberOfSample*2,replace=False,random_state=random_state)
                            else:
                                sample=resample(set_,n_samples=len(set_),replace=False,random_state=random_state)
                          else:
                            sample=resample(set_,n_samples=numberOfSample,replace=False,random_state=random_state)
                       else:
                          sample=resample(set_,n_samples=numberOfSample,replace=False,random_state=random_state)
                    sample=sample.sort_values(['TIMESTAMP'])
                    if i==0 and j==2:
                       accumulator=sample
                    else:
                       accumulator=pd.concat([accumulator,sample],axis=0)
                print("-------> "+str(j)+" hour is complete")
                print("Data length is: {}\n".format( len(accumulator)  ))
            else:
                if(proportion>0):
                   length=int(len(set_)*proportion)
                   if(double_hour==True):
                      if((j==3) or (j==8) or (j==14) or (j==17)):
                         if(length*2<len(set_)):
                            sample=resample(set_,n_samples=length*2,replace=False,random_state=random_state)
                         else:
                            sample=resample(set_,n_samples=len(set_),replace=False,random_state=random_state)
                      else:
                          sample=resample(set_,n_samples=length,replace=False,random_state=random_state)
                   else:
                       sample=resample(set_,n_samples=length,replace=False,random_state=random_state)
                else:
                  break
                sample=sample.sort_values(['TIMESTAMP'])
                if i==0 and j==2:
                  accumulator=sample
                else:
                  accumulator=pd.concat([accumulator,sample],axis=0)
                print("-------> "+str(j)+" hour is complete")
                print("Data length is: {}\n".format( len(accumulator)  ))
    return accumulator
            
    


# In[34]:

import numpy as np
sampling_data=sampling_fliter_machine(df,proportion=0.2,double_hour=True,random_state=0)


# In[30]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt;
plt.title('Travel time, if ceiling is 10000, get logarithm')
plt.hist(np.log(sampling_data['POLYLINE_time_second']),bins=10,normed=True)
plt.xlabel('POLYLINE_time_second(log)')


# In[31]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt;
plt.title('Travel time, if ceiling is 10000')
plt.hist(sampling_data['POLYLINE_time_second'],bins=10,normed=True)
plt.xlabel('POLYLINE_time_second')



%matplotlib inline
import matplotlib.pyplot as plt;
plt.title('Travel time, if ceiling is 10000')
plt.hist(sampling_data['hour_in_day'],bins=10,normed=True)
plt.xlabel('POLYLINE_time_second')

