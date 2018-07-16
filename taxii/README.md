Taxi travel time prediction:

https://www.kaggle.com/c/pkdd-15-taxi-trip-time-prediction-ii

***
# Complete Data handling
## Data preprocessing:
### About IDs:
1. label encode the TAXI ID, to let the number will be the smaller and shorter bit, so it will be easier for training and save the training time.
2. Drop trip ID, because it just labels every trips, so each trip will have different label, and it not has special meaning for the training model, because
About time:
The main goal is to get different type of time day, especially the type is like a cycle, for example, which day in a month, which hour in a day, etc. They will let us easier to train the model, because model of machine learning are not good to find the unseen data value before, which will make them hard to generalize, and transferring to cycle time data type will let future data have some similarity with historical data.
1. Find the based unix timestamp for 12:00am UTC 01/07/2013, and use TIMESTAMP to minus it, then we can get the seconds in a year( from 01/07/2013 to 30/06/2014 ).
2. Values obtained in 1 are divided by 86400( one day’s sum of second), and transfer to integer, get which day in a year. New feature: day_of_year.
3. Values obtained in 1 are divided by 86400*30( one month’s sum of second), and transfer to integer, get which day in a year. New feature: month.
4. Use the values got in 2 and 3, day_of_year minus month*30, get the day in a month. New feature: day_of_month.
5. Values obtained in 4 mod 7 then add one, get the day in a weak.
New feature: day_of_week.
6. Make: (TIMESTAMP - month*30*86400-day_of_month*86400)/ (60*60), it means we get the remainder of seconds in a day, and we divide it with seconds in a hour, after that transfer to integer value. Thus, we get the hour in a day. New feature: hour_in_day.
7. Use the values which are computed from above minus the value of hour_in_day, get the remainder of seconds in a hour, and we divide it by 60, get the minutes in a hour. New feature: minute_in_hour.
 
### About position:
1. Get the data from the POLYLINE and create four features: start
point’s longitude, start point’s latitude, end point’s longitude,
end point’s latitude.
2. Do the distance computation between the start point and end point, using Haversine formula to compute ( It consider the radian of the earth’s surface better than using traditional distance computation, like Euclidean distance, in this data set). New feature: Haversine distance.
About target value(travel time):
1. See how many points traverse in a trip, compute sum of them and minus one, after that multiply 15 seconds( just as the statement on kaggle). The data which does not have any point in the trip get zero.
Target value: POLYLINE_time_second.
2. Another Target value: POLYLINE_time_second_log, Because the distribution of above value is very skewed, so I get the log of them, and the result is like normal distribution, it’s really good! ! The data which does not have any point in the trip get zero.
3. Compute the travel path pass the "hot point" or not


### About other features:
1. Let our model which type of data it is(numerical or categorical), so I impute the missing value of ORIGIN_STAND, and ORIGIN_CALL with -1, and using label encoder to encode them( It encode them by frequency, that is, the most frequent label got 0, second got 1.. ...)
2. Get the dummy variable of CALL_TYPE, for training our Neural Network model better, and replace original CALL_TYPE.
Feature choosing:
1. Drop the zero variance feature, including DAY_TYPE, “MISSING_DATA”.
2. Drop the feature that is always different in the data set, which means the feature values have the number of kinds equal to the length of data set, including TRIP_ID.
3. Using correlation plot to choose which features about “time” to use. Since the features about time is made from TIMESTAMP. Through the process is easily to get the collinearity effect(Sometime will make model overfitting, or bad performance). Thus, I get rid of highly correlated data, like month, TIMESTAMP, and day_of_year.

## Data choosing:
1. Choosing the data that its “MISSING_DATA" is false.  
2. Choosing the data that have start point and end point.
Target value choosing:
*Choose POLYLINE_time_second_log for training model, after model
training, I’ll take exponential value to transfer back for comparison with POLYLINE_time_second.




Detailed writting report : NTUST_MLclass_final.pdf

(on github, pdf link will not work, it needs to be downloaded)

** WARNING: IF YOU EXPAND THE TIME WINDOE, I WOULD NOT RECOMMAND YOU TO USE Bi-directional RNN, IT VIOLATE THE INTUITION!!


