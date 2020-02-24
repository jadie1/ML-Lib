import numpy as np 
import os

stopping_threshold = 1*(10**-5)

def getError(w, inputs, labels):
    summation = 0
    for index in range(inputs.shape[0]):
        label = labels[index]
        prediction = np.dot(w,inputs[index])
        summation += (label - prediction)**2
    return(0.5*summation)

def getGradient(w, inputs, labels):
    gradient_vector = []
    for j in range(inputs.shape[1]):
        summation = 0
        for i in range(inputs.shape[0]):
            label = labels[i]
            prediction = np.dot(w,inputs[i])
            sample = inputs[i][j]
            summation -= (label-prediction)*sample
        gradient_vector.append(summation)
    return(np.array(gradient_vector))

def batchGradientDescent(learning_rate, data,labels):
    w = np.zeros(data.shape[1] + 1)
    inputs = np.ones((data.shape[0], data.shape[1]+1))
    inputs[:,:-1] = data
    w_difference = 1
    epoch_index = 0
    labels = labels.astype('float64')
    error = getError(w,inputs,labels)
    # print(epoch_index, error)
    while w_difference > stopping_threshold:
        epoch_index += 1
        gradient = getGradient(w,inputs,labels)
        w_new = w - (learning_rate*gradient)
        w_difference = np.mean(np.abs(w-w_new))
        w = w_new
        error = getError(w,inputs,labels)
        # if epoch_index % 100 == 0:
            # print(epoch_index, error)
    # print(epoch_index, error)
    return w

def stochasticGradientDescent(learning_rate,data,labels):
    w = np.zeros(data.shape[1] + 1)
    inputs = np.ones((data.shape[0], data.shape[1]+1))
    inputs[:,:-1] = data
    labels = labels.astype('float64')
    error = getError(w,inputs,labels)
    # print(error)
    w_difference = 1
    while w_difference > stopping_threshold:
        for index in range(inputs.shape[0]):
            if w_difference > stopping_threshold:
                data = np.array([inputs[index]])
                label = np.array([labels[index]])
                gradient = getGradient(w,data,label)
                w_new = w - (learning_rate*gradient)
                w_difference = np.mean(np.abs(w-w_new))
                w = w_new
                error = getError(w,inputs,labels)
                # print(error)
    return(w)


def test(w, data, labels):
    inputs = np.ones((data.shape[0], data.shape[1]+1))
    inputs[:,:-1] = data
    labels = labels.astype('float64')
    return(getError(w, inputs, labels))
