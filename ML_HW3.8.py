# η=0.01 and Times = 2000
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

def dataloader(filename):
    data = np.genfromtxt(filename)
    row, col = data.shape
    #add 1 at the front of the x
    X = np.c_[np.ones((row, 1)), data[:, 0: col-1]]
    #Y = data[:, col-1:col]
    Y = data[:,-1]
    return X,Y

def sigmoid(s):
    return  1/(math.exp(-s)+1)

def gradient_descent(X_train, Y_train, X_test, Y_test, lr=0.01, times=2000):
    row, col = X_train.shape
    w = np.zeros(col)
    errors = []
    for _ in range(times):
        gradients = np.zeros(col)
        for (x, y) in zip(X_train, Y_train):
            temp = -y*(w.dot(x))
            gradient =  (-y*x)*sigmoid(temp)
            gradients += gradient
        gradients /= row
		#gradient descent
        w -= gradients*lr
        y_pred = X_test.dot(w)
        err = (np.sum(np.sign(y_pred)*Y_test < 0))/X_test.shape[0]
        errors.append(err)
    return errors

def SGD(X_train, Y_train, X_test, Y_test, lr=0.01, times=2000):
    row, col = X_train.shape
    w = np.zeros(col)
    errors = []
    for i in range(times):
        num = i % row
        x, y = X_train[num], Y_train[num]
        temp = -y*(w.dot(x))
        gradient = (-y*x)*sigmoid(temp)
        #gradient descent
        w -= gradient*lr
        y_pred = X_test.dot(w)
        err = (np.sum(np.sign(y_pred)*Y_test < 0))/X_test.shape[0]
        errors.append(err)
    return errors

def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    X_train, Y_train = dataloader(train_file)
    X_test, Y_test = dataloader(test_file)
    time = 2000
    errors_gd_1 = gradient_descent(X_train, Y_train, X_test, Y_test, times=time)
    errors_sgd_1 = SGD(X_train, Y_train, X_test, Y_test, times=time)
    errors_gd_2 = gradient_descent(X_train, Y_train, X_test, Y_test, lr=0.001, times=time)
    errors_sgd_2 = SGD(X_train, Y_train, X_test, Y_test, lr=0.001, times=time)
    plt.subplot(311)
    plt.plot(errors_gd_1, label='gradient descent')
    plt.plot(errors_sgd_1, label='stochastic gradient descent')
    plt.legend(loc='best')
    plt.xlabel('times')
    plt.ylabel('error')
    plt.title('Eout lr = 0.01')

    plt.subplot(313)
    plt.plot(errors_gd_2, label='gradient descent')
    plt.plot(errors_sgd_2, label='stochastic gradient descent')
    plt.legend(loc='best')
    plt.xlabel('times')
    plt.ylabel('error')
    plt.title('Eout lr = 0.001')
    plt.savefig('hw3_8.png')

if __name__ == '__main__':
	main()

