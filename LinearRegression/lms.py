import numpy as np
from numpy.linalg import norm


class LMS:

    def __init__(self):
        self.weights = None
        self.y = None
        self.threshold = .005

    def get_weights(self):
        return self.weights

    def predict(self, new_data):
        bias_column = np.ones(shape=(len(new_data), 1))
        x = np.concatenate(bias_column, new_data, axis=1)
        x = np.asmatrix(x)
        return np.matmul(x, self.weights)

    def train(self, x, y, gamma, initial_w, t=10, mode="batch", bias=True):



        # append column of 1's
        if bias:
            bias_column = np.ones(shape=(len(x), 1))
            x = np.concatenate((bias_column, x), axis=1)

        # initialize weight vector to zero vector as per assignment
        self.weights = initial_w.astype(float)

        if mode == "batch":
            self.batch_descent(x, y, t, gamma)

        elif mode == "stochastic":
            self.stochastic_descent(x, y, gamma)

    def batch_descent(self, x, y, t, gamma):

        for i in range(t):
            # get the cost of current weight vector and gradient
            cost = self.cost(self.weights, x, y)
            print(cost)
            cost_gradient = self.cost_gradient(self.weights, x, y)

            # if the magnitude of the gradient is small enough, stop interation
            if norm(cost_gradient, 2) < self.threshold:
                break

            # update weight vector
            self.weights -= cost_gradient * gamma

    def stochastic_descent(self, x, y, gamma):

        for i in range(len(x)):
            # get cost and gradient for single point in x,y
            cost = self.cost(self.weights, x, y)
            cost_gradient = self.cost_gradient(self.weights, [x[i]], [y[i]])

            # update weights
            self.weights -= gamma * cost_gradient

            print(cost, " & ", cost_gradient, " & ", self.weights)

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
                yhat = np.dot(weights, x[i])
                gradient[j] += -1 * (y[i] - yhat) * x[i][j]

        return gradient
