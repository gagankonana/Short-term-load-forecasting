
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt

import statistics
import math
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA


def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix

def dwt(a):
    [ca, cd] = pywt.dwt(a,'haar')
    return ca,cd

def idwt(ca,cd):
    ori = pywt.idwt(ca,cd,'haar')
    return ori

def generateData(sample, outputnum):
    a = np.array(sample)
    mu = np.mean(a)
    
    sigma_2 = np.var(a) / 24 
    result = np.random.normal(loc = mu, scale = np.sqrt(sigma_2), size = outputnum) 
    
    
    print('mu = %f\tsigma^2 = %f'%(mu,sigma_2))
    return mu,sigma_2,result

def drawResult(mu,sigma_2,result):
    plt.figure(figsize=(10,8),dpi=80)
    count, bins, ignored = plt.hist(result, 30, normed=True)
    plt.plot(bins, 1/(np.sqrt(2 * np.pi * sigma_2)) *np.exp( - (bins - mu)**2 / (2 * sigma_2) ),linewidth=2, color='r')

def dataset(matrix_load,train_row):
    matrix_load = np.array(matrix_load)
    print("Data shape: ", matrix_load.shape)
    
    train_set = matrix_load[:train_row, :]
    
    np.random.seed(1234)
    
    np.random.shuffle(train_set)
    
    X_train = train_set[:, :-1]
    y_train = train_set[:, -1]
    
    X_test = matrix_load[train_row:, :-1]
    y_test = matrix_load[train_row:, -1]
    nt=int(input("Enter the number of hours to be used to predict the load :"))
    X_test=X_test[:nt]
    y_test=y_test[:nt]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    print(np.shape(X_train), np.shape(X_test))
    return X_train,y_train,X_test,y_test


df_raw = pd.read_csv('../data/ENTSO-E/load.csv', header=0, usecols=[0,1])

df_raw_array = df_raw.values

list_hourly_load = [df_raw_array[i,1]/1000 for i in range(1, len(df_raw))]
print ("Data shape of list_hourly_load: ", np.shape(list_hourly_load))

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
print(k)

list_hourly_load = np.array(list_hourly_load)
shifted_value = list_hourly_load.mean()
list_hourly_load -= shifted_value

a2 , d2 , d1 = pywt.wavedec(list_hourly_load[:-166 * 24], 'db4', mode = 'sym', level = 2)




order_a2 = [3, 2] 
order_d2 = [4, 1, 2] 
order_d1 = [4, 1, 2]
print(order_a2,order_d2,order_d1)

model_a2 = ARMA(a2, order = order_a2)
model_d2 = ARIMA(d2, order = order_d2)
model_d1 = ARIMA(d1, order = order_d1)
result_a2 = model_a2.fit()
result_d2 = model_d2.fit()
result_d1 = model_d1.fit()

plt.figure(figsize=(10,15))
plt.subplot(3,1,1)
plt.plot(a2,'blue')
plt.plot(result_a2.fittedvalues,'red')
plt.title('model_a2')
plt.subplot(3,1,2)
plt.plot(d2,'blue')
plt.plot(result_d2.fittedvalues,'red')
plt.title('model_d2')
plt.subplot(3,1,3)
plt.plot(d1,'blue')
plt.plot(result_d1.fittedvalues,'red')
plt.title('model_d1')
plt.show()

a2_all , d2_all , d1_all = pywt.wavedec(list_hourly_load, 'db4', mode = 'sym', level = 2)
delta = [len(a2_all) - len(a2), len(d2_all) - len(d2), len(d1_all) - len(d1)]
print(delta)

pa2 = model_a2.predict(params = result_a2.params, start = 1, end = len(a2) + delta[0])
pd2 = model_d2.predict(params = result_d2.params, start = 1, end = len(d2) + delta[1])
pd1 = model_d1.predict(params = result_d1.params, start = 1, end = len(d1) + delta[2])

predict_values = pywt.waverec([pa2, pd2, pd1], 'db4')
print(np.shape(predict_values))

plt.figure(figsize=(15,5))
plt.plot(list_hourly_load,label="$true$",c='green')
plt.plot(predict_values,label="$predict$",c='red')
plt.show()


print(len(list_hourly_load),len(predict_values))
mape = statistics.mape((list_hourly_load+shifted_value)*1000,(predict_values+shifted_value)*1000)
print('MAPE is ', mape)
mae = statistics.mae((list_hourly_load+shifted_value)*1000,(predict_values+shifted_value)*1000)
print('MAE is ', mae)
mse = statistics.meanSquareError((list_hourly_load+shifted_value)*1000,(predict_values+shifted_value)*1000)
print('MSE is ', mse)
rmse = math.sqrt(mse)
print('RMSE is ', rmse)
nrmse = statistics.normRmse((list_hourly_load+shifted_value)*1000,(predict_values+shifted_value)*1000)
print('NRMSE is ', nrmse)