import numpy as np
from sklearn.utils import shuffle
import math


def sign(x):
    if x > 0:
        return 1
    else:
        return -1


class Perceptron:

    def __init__(self, gamma, mode="standard"):
        self.mode = mode
        self.gamma = gamma
        self.sign = np.vectorize(sign)
        self.y = None
        self.x = None
        self.w = None

    def train(self, x, y, t=10):
        # append the bias column
        self.x = self.append_b(x)
        self.y = y

        # initialize the weight vector as all zeros
        n = np.shape(self.x)[1]
        weights = np.zeros(n)

        if self.mode == "standard":
            self.w = self.standard_perceptron(weights, t)
        elif self.mode == "voted":
            self.w = self.voted_perceptron(weights, t)
        elif self.mode == "average":
            self.w = self.avg_perceptron(weights, t)
        else:
            print("invalid mode selected")
            return

    def standard_perceptron(self, w, t):
        weights = w
        # standard algorithm
        for epoch in range(t):
            # shuffle the data
            x, y = shuffle(self.x, self.y)
            # for each data point update weight vector
            for i in range(len(x)):
                indicator = y[i] * np.dot(weights, x[i])
                if indicator <= 0:
                    weights = weights + self.gamma * y[i] * x[i]

            return weights

    def voted_perceptron(self, w, t):
        # intitialize variables
        x, y = self.x, self.y
        weights = w
        m = 0
        c = 1
        wc = []
        for epoch in range(t):
            for i in range(len(x)):
                indicator = y[i] * np.dot(weights, x[i])
                if indicator <= 0:
                    # log Wm and Cm
                    wc.append([weights, c])
                    # update weights and reset tracking variables
                    weights = weights + self.gamma * y[i] * x[i]
                    m += 1
                    c = 1
                else:
                    c += 1
        return wc

    def avg_perceptron(self, w, t):
        # initialize
        weights = w
        x, y = self.x, self.y
        a = np.zeros(len(w))
        # same process as standard
        for epoch in range(t):
            for i in range(len(x)):
                indicator = y[i] * np.dot(weights, x[i])
                if indicator <= 0:
                    weights = weights + self.gamma * y[i] * x[i]
                # update a
                a += weights
        return a

    def predict(self, new_x):
        data_matrix = self.append_b(new_x)
       # voted is the only mode with different prediction protocal
        if self.mode == "voted":
            # put w's and c's in matrix and vector for simple computation
            w_matrix = []
            c_vec = []
            for item in self.w:
                w_matrix.append(item[0])
                c_vec.append(item[1])
            # take transpose so each column is a set of weights
            w_matrix = np.matrix(w_matrix).T
            c_vec = np.array(c_vec)

            # get XW
            pred_matrix = np.matmul(data_matrix, w_matrix)
            # get the sign of each entry
            pred_matrix = self.sign(pred_matrix)
            # right multiply by c_vec to get weighted predictions
            weighted_prediction_sums = np.matmul(pred_matrix, c_vec)
            # take the final sign to get final predictions
            return self.sign(np.squeeze(np.array(weighted_prediction_sums)))
        # avg and standard have same prediction protocal
        else:
            pred = np.matmul(data_matrix, self.w)
            return self.sign(pred)


    @staticmethod
    def append_b(matrix):
        m = len(matrix)
        bias_column = np.ones((m, 1))
        return np.concatenate((bias_column, matrix), axis=1)
