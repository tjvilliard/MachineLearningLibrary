import numpy as np
from numpy.linalg import norm
from numpy.random import choice
import math

class LMS:

    def __init__(self):
        self.prev_w = None
        self.weights = None
        self.y = None
        self.threshold = .00001
        self.cost_data = None

    def get_weights(self):
        return self.weights


    def predict(self, new_data):
        bias_column = np.ones(shape=(len(new_data), 1))
        x = np.concatenate(bias_column, new_data, axis=1)
        x = np.asmatrix(x)
        return np.matmul(x, self.weights)

    def train(self, x, y, gamma, initial_w, t=100, mode="batch", bias=True):
        self.cost_data = []
        # append column of 1's
        if bias:
            x = self.append_b(x)

        # initialize weight vector to zero vector as per assignment
        self.weights = initial_w.astype(float)

        if mode == "batch":
            self.batch_descent(x, y, t, gamma)

        elif mode == "stochastic":
            self.stochastic_descent(x, y, t, gamma)

    def batch_descent(self, x, y, t, gamma):
        self.prev_w = np.copy(self.weights)
        for i in range(t):
            # get the cost of current weight vector and gradient
            self.cost_data.append(self.cost(self.weights, x, y))
            cost_gradient = self.cost_gradient(self.weights, x, y)

            # update weight vector
            self.weights -= cost_gradient * gamma

            # check for convergence
            diff_norm = norm(self.weights - self.prev_w, 2)
            if diff_norm > 10**6:
                print("weight vector diverging: lr= ", gamma)
                return
            if diff_norm < self.threshold:
                print("converged: lr= ", gamma, " t= ", i)
                return
            else:
                self.prev_w = np.copy(self.weights)
        print("did not converge: lr= ", gamma)

    def stochastic_descent(self, x, y, t, gamma):
        self.cost_data =[]
        for i in range(t):
            rand_select = choice(range(len(x)), 1)
            # get cost and gradient for single point in x,y
            self.cost_data.append(self.cost(self.weights, x, y))
            cost_gradient = self.cost_gradient(self.weights, x[rand_select], y[rand_select])

            # update weights
            self.weights -= gamma * cost_gradient

            if i >= 1:
                if abs(self.cost_data[i] - self.cost_data[i-1]) < .000001:
                    print("descent converges: t = ", i)
                    return

    @staticmethod
    def solve_weights(x, y, bias=True):
        if len(x) != len(y):
            print("lengths of x and y do not match")
            return
        # append column of ones for bias
        if bias:
            bias_column = np.ones(shape=(len(x), 1))
            x = np.concatenate((bias_column, x), axis=1)

        # special definition for analytical solution
        y.reshape((len(x), 1))
        x = np.asmatrix(x).T


        xxT = np.matmul(x, x.T)
        inv = np.linalg.inv(xxT)
        intermediate = np.matmul(inv, x)
        return np.matmul(intermediate, y)

    @staticmethod
    def cost(weights, x, y):
        """Cost function: Least mean squares"""
        # find sse
        sse = 0
        for i in range(len(x)):
            yhat = weights.dot(np.squeeze(x[i]))
            sse += np.power((y[i] - yhat), 2)
        return .5 * sse

    @staticmethod
    def cost_gradient(weights, x, y):
        gradient = np.zeros(len(weights))
        for j in range(len(weights)):
            for i in range(len(y)):
                yhat = np.dot(weights, np.squeeze(x[i]))
                gradient[j] += -1 * (y[i] - yhat) * x[i][j]

        return gradient

    @staticmethod
    def append_b(matrix):
        m = len(matrix)
        bias_column = np.ones((m, 1))
        return np.concatenate((bias_column, matrix), axis=1)
