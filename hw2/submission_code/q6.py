import numpy as np
import scipy.io as sio
from PIL import Image
from numpy import linalg as LA
import matplotlib.pyplot as plt

# load data

mat_contents = sio.loadmat("mnist_digits.mat") # mat_contents['X'].shape = (10000, 784)
dataset = np.concatenate((mat_contents['X'], mat_contents['Y']), axis=1) # X, Y = dataset[:,:-1], dataset[:,-1]

# split data

def split(dataset, ratio):
    msk = np.random.rand(len(dataset)) < ratio
    return dataset[msk], dataset[~msk]

train_data, test_data = split(dataset, 0.7)

# build perceptron

def perceptr on_train(train_data, max_iter, digit, mode=3):

    if len(train_data)==0: return [[]]
    w = np.zeros((len(train_data[0]), 1)) # initialize weights and bias (index = 0)

    if mode == 0: # Perceptron V0

        for i in range(max_iter):

            datapoint = train_data[i%len(train_data)]
            y = 1 if (datapoint[-1] == digit) else -1
            x = datapoint[:-1]

            if y*x.dot(w[1:])[0] + w[0][0] <= 0:

                w[1:] = w[1:] + y*x.reshape(784,1)
                w[0] = w[0] + y

        return w

    if mode == 1: # Perceptron V1

        for i in range(max_iter):

            datapoint_min = train_data[0]
            y_min = 1 if (datapoint_min[-1] == digit) else -1
            x_min = datapoint_min[:-1]
            a_min = y_min*x_min.dot(w[1:])[0] + w[0][0]

            for datapoint in train_data:

                y = 1 if (datapoint[-1] == digit) else -1
                x = datapoint[:-1]
                a = y*x.dot(w[1:])[0] + w[0][0]

                if a < a_min:
                    datapoint_min = datapoint
                    y_min = y
                    x_min = x
                    a_min = a

            if a_min <= 0:
                w[1:] = w[1:] + y_min * x_min.reshape(784,1)
                w[0] = w[0] + y_min

        return w


    else:

        for i in range(max_iter):

            for datapoint in train_data:

                y = 1 if (datapoint[-1] == digit) else -1
                x = datapoint[:-1]

                if y*x.dot(w[1:])[0] + w[0][0] <= 0:

                    w[1:] = w[1:] + y*x.reshape(784,1)
                    w[0] = w[0] + y

        return w



def perceptron_test(test_datapoint, w):

    return np.sign(test_datapoint[:-1].dot(w[1:])[0] + w[0][0])


def perceptron_accuracy(train_data, test_data, max_iter, digit, mode=3):

    count = 0
    total = len(test_data)
    if total==0: return 0
    w = perceptron_train(train_data, max_iter, digit, mode)

    for datapoint in test_data:

        if (perceptron_test(datapoint, w) == 1 and datapoint[-1]==digit) \
        or (perceptron_test(datapoint, w) == -1 and datapoint[-1]!=digit):
            count += 1

    return count/total


# build perceptron V2

def perceptron_train2(train_data, max_iter, digit):

    if len(train_data)==0: return [[]]

    w = np.zeros((len(train_data[0]), 1)) # initialize weights and bias (index = 0)
    w_list = [w]
    c = np.zeros(len(train_data[0]))
    k = 1

    for i in range(max_iter):

        datapoint = train_data[ i % (len(train_data)) ]
        y = 1 if (datapoint[-1] == digit) else -1
        x = datapoint[:-1]

        if y*x.dot(w[1:])[0] + w[0][0] <= 0:

            w[1:] = w[1:] + y*x.reshape(784,1)
            w[0] = w[0] + y
            c[k+1] = 1
            k += 1
            w_list.append(w)

        else:

            c[k] += 1

    return w_list, c, k


def perceptron_test2(test_datapoint, w_list, c, k):

    res = 0

    for i in range(k):
        w = w_list[i]
        res += c[i] * np.sign(test_datapoint[:-1].dot(w[1:])[0] + w[0][0])

    return np.sign(res)

def perceptron_accuracy2(train_data, test_data, max_iter, digit):

    count = 0
    total = len(test_data)
    if total==0:
        return 0

    w_list, c, k = perceptron_train2(train_data, max_iter, digit)

    for datapoint in test_data:

        if (perceptron_test2(datapoint, w_list, c, k) == 1 and datapoint[-1]==digit) \
        or (perceptron_test2(datapoint, w_list, c, k) == -1 and datapoint[-1]!=digit):
            count += 1

    return count/total



# build kernel perceptron
def perceptron_ker(train_data, test_data, max_iter, digit, p):

    # build kernel
    n_samples, n_features = train_data.shape
    if n_samples==0 or len(test_data)==0: return 0

    alpha = np.zeros(n_samples)
    y = list(map(lambda x: 1 if x else -1, train_data[:,-1] == digit))
    K = (1+np.dot(train_data, train_data.T))**p

    for i in range(max_iter):
        signs = np.sign(K.dot(np.multiply(alpha,y)))
        indexes = np.where(np.equal(y, signs)==False)
        if indexes:
            alpha[indexes[0]] += 1

    # testing
    y_test = list(map(lambda x: 1 if x else -1, test_data[:,-1] == digit))
    K_test = (1+np.dot(test_data, train_data.T))**p
    signs_test = np.sign(K_test.dot(np.multiply(alpha,y)))
    res = np.sum(np.equal(y_test, signs_test))/len(test_data)

    return res
