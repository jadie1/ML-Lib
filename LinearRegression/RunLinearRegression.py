import os
import numpy as np 
import LinearRegression


def getData(csvFile):
	data = []
	with open(csvFile, 'r') as f:
		for line in f:
			data.append(line.strip().split(','))
	data = np.array(data)
	return data[:,:-1], data[:,-1]

csvFile = 'Data/try.csv'
data, labels = getData(csvFile)

weights = LinearRegression.stochasticGradientDescent(data[1:], labels[1:])
print(weights)