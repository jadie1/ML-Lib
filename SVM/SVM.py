# Jadie Adams
# References:
# https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2
# https://github.com/yiboyang/PRMLPY/tree/master/ch7

import os 
import numpy as np
from scipy.optimize import minimize

def predict(w, x):
    return np.sign(np.dot(w,x))

def test(w, data, labels):
    inputs = np.ones((data.shape[0], data.shape[1]+1))
    inputs[:,:-1] = data
    wrong = 0
    for index in range(inputs.shape[0]):
        if labels[index] != predict(w, inputs[index]):
            wrong += 1
    return((wrong/inputs.shape[0])*100)

def update(C, lr, n, w, x, y):  
    val = (y * np.dot(x, w))
    if val <= 1:
        w = w - lr*w + lr*C*n*y*x
    else:
        w = (1-lr)*w
    return w

def gram(X, k):
    N = len(X)
    K = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = k(X[i], X[j])
    return K

def PrimalSVM(epochs, lr_function, C, data, labels):
    w = np.full((data.shape[1] + 1), 0)
    inputs = np.ones((data.shape[0], data.shape[1]+1))
    inputs[:,:-1] = data
    n = inputs.shape[0]
    for e in range(1,epochs):
        learning_rate = lr_function(e)
        shuffle_indexs = np.arange(inputs.shape[0])
        np.random.shuffle(shuffle_indexs)
        X = inputs[shuffle_indexs]
        Y = labels[shuffle_indexs]
        for index in range(n):
            x = X[index]
            y = Y[index]
            w = update(C, learning_rate, n, w, x, y)
    return w

def predictDual(test, X, t, k, a, b):
    a_times_t = a * t
    y = np.empty(len(test)) 
    for i, s in enumerate(test): 
        kernel_eval = np.array([k(s, x_m) if a_m > 0 else 0 for x_m, a_m in zip(X, a)])
        y[i] = a_times_t.dot(kernel_eval) + b
    return y

def DualSVM(kernel, C, data, labels, test_data, test_labels):
    inputs = data.astype(float)
    test_inputs = test_data.astype(float)
    N = inputs.shape[0]
    K = gram(inputs, kernel)
    yKy = labels * K * labels[:, np.newaxis]
    A = np.vstack((-np.eye(N), np.eye(N)))
    b = np.concatenate((np.zeros(N), C * np.ones(N)))

    # Negative of dual Lagrangian
    def loss(a):
        return -(a.sum() - 0.5 * np.dot(a.T, np.dot(yKy, a)))
    def Jacobian_of_loss(a):
        return np.dot(a.T, yKy) - np.ones_like(a)

    constraints = ({'type': 'ineq', 'fun': lambda x: b - np.dot(A, x), 'jac': lambda x: -A},
                   {'type': 'eq', 'fun': lambda x: np.dot(x, labels), 'jac': lambda x: labels})
    # training
    a0 = np.random.rand(N)  # initial guess
    res = minimize(loss, a0, jac=Jacobian_of_loss, constraints=constraints, method='SLSQP', options={})

    a = res.x  # optimal Lagrange multipliers
    a[np.isclose(a, 0)] = 0  
    a[np.isclose(a, C)] = C  

    support_idx = np.where(0 < a)[0] 
    margin_idx = np.where((0 < a) & (a < C))[0] 

    print("Number of support vectors = " + str(len(support_idx)))

    # bias term
    a_times_label = a * labels
    cum_b = 0
    for n in margin_idx:
        x_n = inputs[n]
        kernel_eval = np.array([kernel(x_m, inputs[n]) if a_m > 0 else 0 for x_m, a_m in zip(inputs, a)])
        b = labels[n] - a_times_label.dot(kernel_eval)
        cum_b += b
    b = cum_b / len(margin_idx)

    weights = np.sum(a[support_idx, None]*labels[support_idx, None]*inputs[support_idx], axis = 0)

    train_predictions = predictDual(inputs, inputs, labels, kernel, a, b)
    num_wrong = np.sum((train_predictions * labels) < 0)
    print('Train error: ' + str(num_wrong / N))
    test_predictions = predictDual(test_inputs, inputs, labels, kernel, a, b)
    num_wrong = np.sum((test_predictions * test_labels) < 0)
    print('Test error: ' + str(num_wrong / test_inputs.shape[0]))
    parameters = np.zeros(weights.shape[0] + 1)
    parameters[:-1] = weights
    parameters[-1] = b
    return parameters