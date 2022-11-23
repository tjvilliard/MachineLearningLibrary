import numpy
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
import math


def sign(x):
    if x > 0:
        return 1
    else:
        return -1


class SVM:

    def __init__(self, c, max_epoch=10, a=1, gamma=0.5, gamma_schedule=1, mode="primal"):

        self.max_epoch = max_epoch
        self.c = c
        self.a = a
        self.gamma_schedule = gamma_schedule
        self.gamma_zero = gamma
        self.mode = mode
        self.w = None
        self.y = None
        self.x = None
        self.a_star =None
        self.b_star =None
        self.support_vector_idxs = []

        self.sign = np.vectorize(sign)

    def train(self, x, y):
        self.x = x
        self.y = y

        if self.mode == "primal":
            # sets the objects weight vector
            self.primal_solution(x, y)

        elif self.mode == "dual" :
            self.dual_solution()

        elif self.mode == "gaussian-kernel":
            self.kernel_solution()

    def primal_solution(self, x, y):
        # number of training example
        n = len(x)

        # w0 and w tracked independently, updated dependently
        w_0 = np.zeros(np.shape(x)[1])
        w_b = np.zeros(np.shape(x)[1] + 1)

        # prepend bias column
        x_b = self.append_b(x)

        for t in range(self.max_epoch):
            # shuffle the data
            x_shuffle, y_shuffle = shuffle(x_b, y)

            # set gamma according to schedule
            gamma = self.gamma_t(t)

            # iterate shuffled data
            for i in range(n):
                # indicator function
                indicator = y_shuffle[i] * np.dot(w_b, x_shuffle[i])

                if indicator <= 1:
                    # get w0 ie set bias to 0 temporarily
                    w_adjust = np.concatenate((np.array([0]), w_0), axis=0)
                    w_b = w_b - (gamma * w_adjust) + (gamma * self.c * n * y_shuffle[i] * x_shuffle[i])
                else:
                    w_0 = (1 - gamma) * w_0

            self.w = w_b

    def dual_solution(self):

        # set up for minimize
        objective = self.dual_form_objective
        c = self.c
        cons = {"type": "eq", "fun": lambda x: np.dot(self.y, x)}
        bnds = Bounds(lb=0, ub=self.c)
        x0 = np.zeros(len(self.x))
        minimize_result = minimize(objective, x0=x0, bounds=bnds, constraints=cons,  method="SLSQP")

        # using optimal langrians, get w* and b*
        m = np.shape(self.x)[0]
        n = np.shape(self.x)[1]
        b_star_list = []
        w_star = np.zeros(n)
        lagrangian = minimize_result.x

        # get w*
        for i in range(m):
            a = lagrangian[i]
            xi = self.x[i]
            yi = self.y[i]
            if True:
                self.support_vector_idxs.append(i)
                w_star += a * yi * xi
        # get average b* using w*
        for i in range(m):
            xi = self.x[i]
            yi = self.y[i]
            bi = yi - np.dot(w_star, xi)
            b_star_list.append(bi)

        b_star = np.mean(b_star_list)
        self.w = np.concatenate((np.array([b_star]), w_star), axis=0)

    def kernel_solution(self):
        # set up for minimize
        objective = self.gaussian_objective
        cons = {"type": "eq", "fun": lambda input_vector: np.dot(self.y, input_vector)}
        bnds = Bounds(lb=0, ub=self.c)
        x0 = np.zeros(len(self.x))
        minimize_result = minimize(objective, x0=x0, bounds=bnds, constraints=cons, method="SLSQP", jac=True)

        # using optimal langrians, get w* and b*
        m = np.shape(self.x)[0]
        n = np.shape(self.x)[1]
        b_star_list = []
        #w_star = np.zeros(n)
        self.a_star = minimize_result.x

        self.support_vector_idxs = np.where(np.invert(np.isclose(self.a_star, np.zeros(m), atol=.00001)))[0]
        # get average b* using w*
        for idx in self.support_vector_idxs:
            x = self.x[idx]
            yi = self.y[idx]
            rolling_sum = 0
            for j in self.support_vector_idxs:
                yk = self.y[j]
                xi = self.x[j]
                a_star = self.a_star[j]
                kernel = self.gaussian_kernel_function(xi, x, self.gamma_zero)
                rolling_sum += a_star * yk * kernel
            bi = yi - rolling_sum
            b_star_list.append(bi)

        self.b_star = np.mean(b_star_list)


    def predict(self, new_x):
        if self.mode == "gaussian-kernel":
            return self.kernel_predict(new_x)

        data_matrix = self.append_b(new_x)
        pred_vec = np.matmul(data_matrix, self.w)
        return self.sign(pred_vec)

    def kernel_predict(self, new_x):
        idxs = self.support_vector_idxs
        support_vectors = self.x[idxs]

        # CREDIT: https://stats.stackexchange.com/questions/15798/how-to-calculate-a-gaussian-kernel-effectively-in-numpy
        pairwise_sq_dist = cdist(new_x, support_vectors, "sqeuclidean")
        K = scipy.exp(-pairwise_sq_dist / self.gamma_zero)

        ay = self.a_star[idxs] * self.y[idxs]
        kay_matrix = np.matmul(K, ay)

        pred_vec = kay_matrix + self.b_star
        return self.sign(pred_vec)

    def gamma_t(self, t):
        if self.gamma_schedule == 1:
            return self.gamma_zero / (1 + self.gamma_zero * t / self.a)
        elif self.gamma_schedule == 2:
            return self.gamma_zero / (1 + t)

    def dual_form_objective(self, alpha):
        x = self.x
        y = self.y

        # element wise mult of a, y, and rows of x
        #CHECK--------------------------------------------------------------
        ay = alpha * y
        xya_matrix = x*ay[:, np.newaxis]

        # entry i,j in matrix is i,j in dual form sum
        expanded_dual_matrix = np.matmul(xya_matrix, xya_matrix.T)

        # sum all rows
        double_sum = sum(np.sum(expanded_dual_matrix, axis=1))
        a_sum = sum(alpha)
        # return sum of sum of rows
        return .5 * double_sum - a_sum

    def gaussian_objective(self, alpha):

        # CREDIT: https://stats.stackexchange.com/questions/15798/how-to-calculate-a-gaussian-kernel-effectively-in-numpy
        pairwise_sq_dist = cdist(self.x, self.x,  "sqeuclidean")
        K = np.exp(-pairwise_sq_dist / self.gamma_zero)

        ay_elements = np.multiply(np.outer(alpha, alpha), np.outer(self.y, self.y))
        kya_matrix = np.multiply(K, ay_elements)

        double_sum = sum(np.sum(kya_matrix, axis=1))
        a_sum = sum(alpha)

        obj_val = 0.5 * double_sum - a_sum

        # compute the gradient wtr to alpha
        ay = np.multiply(alpha, self.y)
        k_apply_ay = np.matmul(K, ay)
        ew_y_kay = np.multiply(self.y, k_apply_ay)
        obj_grad = 0.5 * ew_y_kay - 1
        return obj_val, obj_grad


    def gaussian_kernel_function(self, xi, xj, gamma):
        diff_vec = xi - xj
        mag = np.linalg.norm(diff_vec, 2)
        exponent = -1 * np.power(mag, 2) / gamma
        return np.exp(exponent)

    @staticmethod
    def append_b(matrix):
        m = len(matrix)
        bias_column = np.ones((m, 1))
        return np.concatenate((bias_column, matrix), axis=1)
