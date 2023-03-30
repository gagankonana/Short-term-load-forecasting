from __future__ import print_function
print("importing Libraries")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
import statistics
import math
import os

print("Loading data...")
def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix


df_raw = pd.read_csv('../data/ENTSO-E/load.csv', header=0, usecols=[0,1])
#print("here   ",df_raw[:1])
dates=df_raw.date

df_raw_array = df_raw.values

list_hourly_load = [df_raw_array[i,1]/1000 for i in range(0, len(df_raw))]
#print("Here" ,list_hourly_load)
#print ("Data shape of list_hourly_load: ", np.shape(list_hourly_load))
print("Processing data...")
k = 0
for j in range(0, len(list_hourly_load)):
    if(abs(list_hourly_load[j]-list_hourly_load[j-1])>2 and abs(list_hourly_load[j]-list_hourly_load[j+1])>2):
        k = k + 1
        list_hourly_load[j] = (list_hourly_load[j - 1] + list_hourly_load[j + 1]) / 2 + list_hourly_load[j - 24] - list_hourly_load[j - 24 - 1] / 2
    sum = 0
    num = 0
    for t in range(1,8):
        if(j - 24*t >= 0):
            num = num + 1
            sum = sum + list_hourly_load[j - 24*t]
        if(j + 24*t < len(list_hourly_load)):
            num = num + 1
            sum = sum + list_hourly_load[j + 24*t]
    sum = sum / num
    if(abs(list_hourly_load[j] - sum)>3):
        k = k + 1
        if(list_hourly_load[j] > sum): list_hourly_load[j] = sum + 3
        else: list_hourly_load[j] = sum - 3
#print(k)
#plt.plot(list_hourly_load)
#plt.show()

list_hourly_load = np.array(list_hourly_load)
shifted_value = list_hourly_load.mean()
list_hourly_load -= shifted_value

sequence_length = 25
#print("Here" ,list_hourly_load[:1],sequence_length)
matrix_load = convertSeriesToMatrix(list_hourly_load, sequence_length)
matrix_load = np.array(matrix_load)
#print ("Data shape: ", matrix_load.shape)


train_row = matrix_load.shape[0] - 166*24
#print('train:',train_row,'test:',166*24)
train_set = matrix_load[:train_row, :]

np.random.seed(1234)

np.random.shuffle(train_set)

X_train = train_set[:, :-1]

y_train = train_set[:, -1]
#print(X_train[0],y_train[0])

X_test = matrix_load[train_row:, :-1]
y_test = matrix_load[train_row:, -1]
time_test = [df_raw_array[i,0] for i in range(train_row+23, len(df_raw))]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
mode=int(input("Enter 1 to train the model, Enter 2 to use the existing trained model : "))
if mode==1:
	model = Sequential()
	model.add(LSTM( input_dim=1, units=50, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(100, return_sequences=False))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation='linear'))
	model.summary()
	model.compile(loss="mse", optimizer="rmsprop")
	model.fit(X_train, y_train, batch_size=512, epochs=100, validation_split=0.05, verbose=2)
	model.save('../model/lstm.h5')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('../model/lstm.h5')
ron=int(input("Enter 1 to predict load for given hour and 2 to predict load for a range of hours: "))
if ron==1:
    for i in range(0,10):
        print(i+1,dates[16752+i])
    hn=int(input("Select time from the above list to predict the date :"))
    st=hn
    nt=hn+1
else:
    st=int(input("Enter the range of hours to predict the load\nfrom:"))
    nt=int(input("to:"))
    print("predicting load for hours from",dates[16751+st],"to",dates[16751+nt])

X_test=X_test[st:nt]
y_test=y_test[st:nt]
time_test=time_test[st:nt]
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
test_mse = model.evaluate(X_test, y_test, verbose=2)
predicted_values = model.predict(X_test)
#print(X_test[:1])
num_test_samples = len(predicted_values)
predicted_values = np.reshape(predicted_values, (num_test_samples,1))
#print(predicted_values, y_test)
print("------LSTM--------")
print("      date/time.            |   Predicted values(kW)   |    Actual values(kW)")
for i in range(0,num_test_samples):
	print(dates[16751+st+i],"  ","     ", predicted_values[i]+shifted_value,"               ",y_test[i]+shifted_value)
if ron==2:
	print ('\nThe MSE on the test data set is %.3f for %d test samples.' % (test_mse, len(y_test)))
	mape = statistics.mape((y_test+shifted_value)*1000,(predicted_values+shifted_value)*1000)
	print('MAPE is ', mape)
	mae = statistics.mae((y_test+shifted_value)*1000,(predicted_values+shifted_value)*1000)
	print('MAE is ', mae)
	mse = statistics.meanSquareError((y_test+shifted_value)*1000,(predicted_values+shifted_value)*1000)
	print('MSE is ', mse)
	rmse = math.sqrt(mse)
	print('RMSE is ', rmse)
	nrmse = statistics.normRmse((y_test+shifted_value)*1000,(predicted_values+shifted_value)*1000)
	print('NRMSE is ', nrmse)
	fig = plt.figure()
	plt.plot(y_test + shifted_value, label="$true$", c='green')
	plt.plot(predicted_values + shifted_value, label="$predict$", c='red')
	plt.xlabel('Hour')
	plt.ylabel('Electricity load (*1e3)')
	plt.legend()
	plt.show()
	fig.savefig('../result/lstm_result.jpg', bbox_inches='tight')