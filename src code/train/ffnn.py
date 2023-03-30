print("importing Libraries")
import math
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.decomposition import PCA
def fit_predict(xTrain, yTrain, xTest, epochs, neurons):
    
    if (not len(xTrain) == len(yTrain) or len(xTrain) == 0 or
                len(xTest) == 0 or epochs <= 0):
        return

    
    
    indices = np.arange(len(xTrain))
    np.random.shuffle(indices)

    trainSwapX = [xTrain[x] for x in indices]
    trainSwapY = [yTrain[x] for x in indices]

    supTrain = SupervisedDataSet(len(xTrain[0]), 1)
    for x in range(len(trainSwapX)):
        supTrain.addSample(trainSwapX[x], trainSwapY[x])

    

    n = FeedForwardNetwork()

    inLayer = LinearLayer(len(xTrain[0]))
    hiddenLayer1 = SigmoidLayer(neurons)
    outLayer = LinearLayer(1)

    n.addInputModule(inLayer)
    n.addModule(hiddenLayer1)
    n.addOutputModule(outLayer)

    in_to_hidden = FullConnection(inLayer, hiddenLayer1)
    hidden_to_out = FullConnection(hiddenLayer1, outLayer)

    n.addConnection(in_to_hidden)
    n.addConnection(hidden_to_out)

    n.sortModules()

    
    

    trainer = BackpropTrainer(n, dataset=supTrain, momentum=0.1, learningrate=0.01
                              , verbose=False, weightdecay=0.01)

    trainer.trainUntilConvergence(dataset=supTrain,
                                  maxEpochs=epochs, validationProportion=0.30)

    outputs = []
    for x in xTest:
        outputs.append(n.activate(x))

    return outputs


def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix

print("Loading data...")
df_raw = pd.read_csv('../data/ENTSO-E/load.csv', header=0, usecols=[0,1])

df_raw_array = df_raw.values
dates=df_raw.date

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
    st=int(input("Enter the range of hours to predict the load \nfrom:"))
    nt=int(input("to:"))
    print("predicting load for hours from",dates[16751+st],"to",dates[16751+nt])
X_test=X_test[st:nt]
y_test=y_test[st:nt]
time_test=time_test[st:nt]
num_test_samples=len(y_test)
dimensions = [6,10,12]
neurons = [30,50,50]
names = []
names.append('true')
for x in range(len(dimensions)):
    s = "d=" + str(dimensions[x]) + ",h=" + str(neurons[x])
    names.append(s)
preds = []
preds.append(y_test)
for x in range(len(dimensions)):
    
    pca = PCA(n_components=dimensions[x])
    pca.fit(X_train)
    xTrainRed = pca.transform(X_train)
    xTestRed = pca.transform(X_test)
    predicted_values = fit_predict(xTrainRed, y_train, xTestRed, 1, neurons[x])
    print("------FFNN-------")
    print("with dimensions=",dimensions[x],"and neurons=",neurons[x])
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
        x, = plt.plot(preds[j]+shifted_value, color=colors[j])
        legendVars.append(x)
    plt.xlabel('Hour')
    plt.ylabel('Electricity load (*1e3)')
    plt.legend(legendVars, names)
    plt.show()
    fig.savefig('../result/nn_result.jpg', bbox_inches='tight')