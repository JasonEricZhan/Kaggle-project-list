
import numpy as np
from multiprocessing import Pool
#add to the list
#df there is pandas data frame

hot_point=[[-8.61470,41.14660],[-8.616,41.141],[-8.60658,41.14724],[-8.624817,41.177124],[-8.64927,41.170653],[-8.6382,41.159151],
          [-8.624817,41.177124],[-8.64927,41.170653],[-8.6382,41.159151]]
          
def get_dist(lonlat,point):
    if len(lonlat) >0:
       lon_diff = np.abs(lonlat[0]-point[0])*np.pi/360.0
       lat_diff = np.abs(lonlat[1]-point[1])*np.pi/360.0
       a = np.sin(lat_diff)**2 + np.cos(point[0]*np.pi/180.0) * np.cos(point[1]*np.pi/180.0) * np.sin(lon_diff)**2  
       d = 2*6371*np.arctan2(np.sqrt(a), np.sqrt(1-a))
       return(d)
    else:
       return 0
       
       

def pass_hot_point(lonlat):
    if len(lonlat) >0:
       length=len(lonlat)
       len_hot=len(hot_point)
       for i in range(0,length):
           for j in range(0,len_hot):
               dist=get_dist(lonlat[i],hot_point[j])
               if(dist<0.1):
                     return True
               else:
                    pass
       #if(i%1000==0):
            #print(i)
       return False
    else:
       return False
       
       

num_partitions = 3 #number of partitions to split dataframe
num_cores = 3 #number of cores on the machine

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df=pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def multiply_hot_point(data):
    data['pass_hot_point'] = data['POLYLINE'].apply(lambda x: pass_hot_point(x))
    return data
    
df= parallelize_dataframe(df, multiply_hot_point)


