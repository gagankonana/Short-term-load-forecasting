import os
n=int(input("Enter \n1 for Clustering model\n2 for LSTM model\n3 for ARIMA\n4 for SVR\n5 for DBN\n6 for FFNN\n7 for prediction using all the algorithms:"))
if n==1 or n==7:
	os.system("python clustering.py")
if n==2 or n==7:
	os.system("python lstm.py")
if n==3  or n==7:
	os.system("python arima.py")
if n==4 or n==7:
	os.system("python svr.py")
if n==5 or n==7:
	os.system("python dbn.py")
if n==6 or n==7:
	os.system("python ffnn.py")
