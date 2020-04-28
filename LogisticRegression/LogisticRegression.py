# Jadie Adams
import numpy as np 
import os
from math import exp

def getError(w, data, labels):
    errors = 0
    inputs = np.ones((data.shape[0], data.shape[1]+1))
    inputs[:,:-1] = data
    labels = labels.astype('float64')
    for index in range(inputs.shape[0]):
        prediction = predict(inputs[index], w)
        if prediction >= 0.5:
            prediction =  1 
        else:
            prediction = 0 
        if labels[index] != prediction:
            errors += 1
    return(errors/len(labels))

def predict(x, w):
    return 1.0 / (1.0 + exp(-np.dot(x,w)))

def MLstochasticGradientDescent(lr_function,data,labels):
    w = np.zeros(data.shape[1] + 1)
    inputs = np.ones((data.shape[0], data.shape[1]+1))
    inputs[:,:-1] = data
    for epoch in range(50):
        learning_rate = lr_function(epoch)
        sum_error = 0
        indices = np.arange(10)
        np.random.shuffle(indices)
        inputs = inputs[indices]
        labels = labels[indices]
        for index in range(inputs.shape[0]):
            x = inputs[index]
            y = labels[index]
            prediction = predict(inputs[index],w)
            error = labels[index] - prediction
            sum_error += error**2
            w = w + (learning_rate*error*prediction*(1-prediction)*inputs[index])
        print('epoch=%d, error=%.3f' % (epoch, sum_error))
    return(w)
