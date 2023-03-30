import math
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import euclidean
from sklearn.cluster import SpectralClustering
def kMeansClustering(x,k):
    print("Finiding centroids and labels with iter=10 ")
    conv = np.asarray(x)    
    centroids = kmeans(conv,k,iter=10)[0]    
    labels = []
    for y in range(len(x)):
        minDist = float('inf')
        minLabel = -1
        for z in range(len(centroids)):
            e = euclidean(conv[y],centroids[z]) 
            if (e < minDist):
                minDist = e
                minLabel = z
        labels.append(minLabel)    
    return (centroids,labels)
def predictClustering(clusters,clusterSets,xTest,metric):
    print("Predicting...")
    clustLabels = []
    simFunction = getDistLambda(metric)
    for x in range(len(xTest)):
        clustDex = -1
        clustDist = float('inf')
        for y in range(len(clusters)):
            dist = simFunction(clusters[y],xTest[x])
            if (dist < clustDist):
                clustDist = dist
                clustDex = y
        clustLabels.append(clustDex)
    predict = np.zeros(len(xTest))
    for x in range(len(xTest)):
        predict[x] = weightedClusterClass(xTest[x],clusterSets[clustLabels[x]],simFunction)
    return predict
def weightedClusterClass(xVector,examples,simFunction):
    pred = 0.0
    normalizer = 0.0
    ctr = 0
    for x in examples:
        similarity = 1.0/simFunction(xVector,x[0])
        pred += similarity*x[1]
        normalizer += similarity
        ctr += 1
    return (pred/normalizer)
def getDistLambda(metric):
    if (metric == "manhattan"):
        return lambda x,y : distance.cityblock(x,y)
    elif (metric == "cosine"):
        return lambda x,y : distance.cosine(x,y)
    else:
        return lambda x,y : distance.euclidean(x,y)
def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix
print("Loading data...")
df_raw = pd.read_csv('../data/ENTSO-E/load.csv', header=0, usecols=[0,1])
dates=df_raw.date
df_raw_array = df_raw.values
list_hourly_load = [df_raw_array[i,1]/1000 for i in range(0, len(df_raw))]
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
list_hourly_load = np.array(list_hourly_load)
shifted_value = list_hourly_load.mean()
list_hourly_load -= shifted_value
sequence_length = 25
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
ckmeans_365,lkmeans_365 = kMeansClustering(X_train,365)
c = [ckmeans_365]
l = [lkmeans_365]
algNames = ["true","k-means"]
preds = []
preds.append(y_test)
num_test_samples=nt-st
for t in range(len(c)):
    centroids = c[t]
    labels = l[t]
    clusterSets = []
    timeLabels = []
    for x in range(len(centroids)):
        clusterSets.append([])
    for x in range(len(labels)):
        clusterSets[labels[x]].append((X_train[x], y_train[x]))
    predicted_values = predictClustering(centroids, clusterSets, X_test, "euclidean")
    print("------K-means--------")
    print("      date/time.            |   Predicted values(kW)   |    Actual values(kW)")
    for i in range(0,num_test_samples):
        print("  ",dates[16751+st+i],"     ", predicted_values[i]+shifted_value,"               ",y_test[i]+shifted_value)
    if ron==2:
        mape = statistics.mape((y_test + shifted_value) * 1000, (predicted_values + shifted_value) * 1000)
        print('MAPE is ', mape)
        mae = statistics.mae((y_test + shifted_value) * 1000, (predicted_values + shifted_value) * 1000)
        print('MAE is ', mae)
        mse = statistics.meanSquareError((y_test + shifted_value) * 1000, (predicted_values + shifted_value) * 1000)
        print('MSE is ', mse)
        rmse = math.sqrt(mse)
        print('RMSE is ', rmse)
        nrmse = statistics.normRmse((y_test + shifted_value) * 1000, (predicted_values + shifted_value) * 1000)
        print('NRMSE is ', nrmse)
    preds.append(predicted_values)
if ron==2:
    fig = plt.figure()
    colors = ["g","r","b","c","m","y","k","w"]
    legendVars = []
    for j in range(len(preds)):
        #print(j)
        x, = plt.plot(preds[j]+shifted_value, color=colors[j])
        legendVars.append(x)
    plt.xlabel('Hour')
    plt.ylabel('Electricity load (*1e3)')
    plt.legend(legendVars, algNames)
    plt.show()
    fig.savefig('../result/clustering_result.jpg', bbox_inches='tight')