import numpy as np
import pandas as pd
import torch
from NeuralNetworks.neuralnet_2 import NeuralNet


def main():
    col_names = ["variance", "skewness", "curtosis", "entropy", "label"]

    train = pd.read_csv("Data/train.csv", names=col_names)
    # train["label"] = train["label"].replace([0], -1)
    train = train.to_numpy()

    test = pd.read_csv("Data/test.csv", names=col_names)
    # test["label"] = test["label"].replace([0], -1)
    test = test.to_numpy()

    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]

    x_train = append_b(x_train)
    x_test = append_b(x_test)

    np.random.seed(10)

    d, lr = 10, .025

    widths = [5, 10, 25, 50, 100]
    num_inputs = np.shape(x_train)[1]
    for width in widths:
        print("hidden layer width = ", width, "  =========================================")
        net_structure = (num_inputs, width, width, 1)
        nn_init_zero = NeuralNet(net_structure, init_zero=True)
        nn_init_random = NeuralNet(net_structure, init_zero=False)

        print("Training init_zero")
        nn_init_zero.train(x_train, y_train, d=d, learning_rate=lr)
        print(" \n")

        init_zero_pred_train = nn_init_zero.predict(x_train)
        init_zero_pred_test = nn_init_zero.predict(x_test)

        print("Training init_random")
        nn_init_random.train(x_train, y_train, d=d, learning_rate=lr)
        #print(nn_init_random.training_y_hat[0:20])
        print("\n ")

        init_rand_pred_train = nn_init_random.predict(x_train)
        init_rand_pred_test = nn_init_random.predict(x_test)

        print("init_random Train Error: ", prediction_error(y_train, init_rand_pred_train))
        print("init_random Test Error: ", prediction_error(y_test, init_rand_pred_test))
        print('\n')
        print("init_zero Train Error: ", prediction_error(y_train, init_zero_pred_train))
        print("init_zero Test Error: ", prediction_error(y_test, init_zero_pred_test))
        print("==============================================================================================")









def paper_problem_scratch():
    w = np.array([0.0372, 0.0148, 0.0015])

    x = np.array([[.5, -1, 0.3],
                  [-1, -2, -2],
                  [1.5, 0.2, -2.5]])
    y = np.array([1, -1, 1])

    gradient = np.zeros(3)
    for j in range(3):
        for i in range(3):
            gradient[j] += -y[i] * x[i][j] / (1 + np.exp(y[i] * w.dot(x[i])))

        gradient[j] += w[j]

    new_w = w - .0025 * gradient
    print(gradient)
    print(new_w)


def append_b(matrix):
    m = len(matrix)
    bias_column = np.ones((m, 1))
    return np.concatenate((bias_column, matrix), axis=1)


def prediction_error(actual, prediction):
    if len(actual) != len(prediction):
        print("arrays not equal length")
        return None
    error = np.equal(actual, prediction)
    error_rate = np.round(1 - (sum(error) / len(actual)), 4)
    return error_rate

if __name__ == "__main__":
    main()