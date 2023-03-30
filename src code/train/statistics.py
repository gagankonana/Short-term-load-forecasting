import numpy as np
import math
import scipy.stats as stats


def meanSquareError(actual,pred):
	if (not len(actual) == len(pred) or len(actual) == 0):
		return -1.0
	total = 0.0
	for x in range(len(actual)):
		total += math.pow(actual[x]-pred[x],2)
	return total/len(actual)
def mse(actual,pred):
	if (not len(actual) == len(pred) or len(actual) == 0):
		return -1.0
	total = 0.0
	for x in range(len(actual)):
		total += math.pow(actual[x]-pred[x],2)
	return total/(len(actual)*1000000)
def normRmse(actual,pred):
	if (not len(actual) == len(pred) or len(actual) == 0):
		return -1.0
	sumSquares = 0.0
	maxY = actual[0]
	minY = actual[0]
	for x in range(len(actual)):
		sumSquares += math.pow(pred[x]-actual[x],2.0)
		maxY = max(maxY,actual[x])
		minY = min(minY,actual[x])
	return math.sqrt(sumSquares/len(actual))/(maxY-minY)
def Rmse(actual,pred):
	if (not len(actual) == len(pred) or len(actual) == 0):
		return -1.0
	sumSquares = 0.0
	for x in range(len(actual)):
		sumSquares += math.pow(pred[x]-actual[x],2.0)
	return math.sqrt(sumSquares/len(actual))
def mape(actual,pred):
	if (not len(actual) == len(pred) or len(actual) == 0):
		return -1.0
	total = 0.0
	for x in range(len(actual)):
		total += abs((actual[x]-pred[x])/actual[x])
	return total/len(actual)
def mae(actual,pred):
	if (not len(actual) == len(pred) or len(actual) == 0):
		return -1.0
	total = 0.0
	for x in range(len(actual)):
		total += abs(actual[x]-pred[x])
	return total/len(actual)