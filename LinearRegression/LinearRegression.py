import numpy as np 


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

def batchGradientDescent(data,labels):
    threshold = 1
    learning_rate = 0.01
    # initialize weights
    w = np.array([0,0,0,0]).astype('float64')
    # add 1 to data vectors for b
    inputs = np.ones((data.shape[0], data.shape[1]+1))
    inputs[:,:-1] = data
    labels = labels.astype('float64')
    # get initial error
    error = getError(w,inputs,labels)
    while error > threshold:
        gradient = getGradient(w,inputs,labels)
        w = w - (learning_rate*gradient)
        error = getError(w,inputs,labels)
    return w

def stochasticGradientDescent(data,labels):
    threshold = 1*(10**-6)
    learning_rate = 0.1
    # initialize weights
    w = np.array([0,0,0,0]).astype('float64')
    # add 1 to data vectors for b
    inputs = np.ones((data.shape[0], data.shape[1]+1))
    inputs[:,:-1] = data
    labels = labels.astype('float64')
    # get initial error
    error = getError(w,inputs,labels)
    print(error)
    w_difference = 1
    while w_difference > threshold:
        for index in range(inputs.shape[0]):
            if w_difference > threshold:
                data = np.array([inputs[index]])
                label = np.array([labels[index]])
                gradient = getGradient(w,data,label)
                w_new = w - (learning_rate*gradient)
                w_difference = np.mean(np.abs(w-w_new))
                w = w_new
                error = getError(w,inputs,labels)
                print(error)
    return(w)