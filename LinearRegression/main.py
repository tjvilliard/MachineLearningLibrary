import pandas as pd
import numpy as np
from lms import LMS
import matplotlib.pyplot as plt

def main():

    #paper_problem_work()

    assignment_a()
    assignment_b()


def paper_problem_work():
    w = np.array([0, 0, 0, 0])
    w1 = np.array([-1, -1, 1, -1])
    y = np.array([1, 4, -1, -2, 0])
    x = np.array([[1, -1, 2],
                  [1, 1, 3],
                  [-1, 1, 0],
                  [1, 2, -4],
                  [3, -1, 1]])

    lms = LMS()
    lms.train(x, y, initial_w=w, gamma=0.1, mode="stochastic")
    print(lms.weights)
    print("\n")
    lms.train(x, y, initial_w=w1, gamma=0.1, mode="batch")
    print(lms.weights)
    w_star = lms.solve_weights(x, y)
    bias_column = np.ones(shape=(len(x), 1))
    x_b = np.concatenate((bias_column, x), axis=1)
    min_cost = lms.cost(w_star, x_b, y)
    print(w_star, " ", min_cost)


def assignment_a():
    # concrete data
    col_names = range(8)
    train = pd.read_csv("Data/train.csv", names=col_names).to_numpy()
    test = pd.read_csv("Data/test.csv", names=col_names).to_numpy()

    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]

    w_i = np.zeros(8)
    lms = LMS()

    run = False
    if run:
        # Determined the optimal learning rate: 0.014
        gamma_set1 = [1, .5, .25, .125, .0625, .03125, .015, 0.01, .0075]
        gamma_set2 = np.arange(0.008, 0.02, .001)
        for gamma in list(gamma_set2):
            print(gamma)
            lms.train(x_train, y_train, gamma, initial_w=w_i, mode="batch", t=10000)
            print("\n")

    # display data
    lms.train(x_train, y_train, gamma=0.014, initial_w=w_i, mode="batch", t=10000)

    plt.plot(range(len(lms.cost_data)), lms.cost_data)
    plt.title("Lms Cost at step t: gamma = 0.014")
    plt.xlabel("step t")
    plt.savefig("Experiments/lms_data_a.png")
    print("learned weights: ", lms.weights)
    print(lms.solve_weights(x_train, y_train))
    x_test_b = lms.append_b(x_test)
    print(lms.cost(lms.weights, x_test_b, y_test))

def assignment_b():
    # concrete data
    col_names = range(8)
    train = pd.read_csv("Data/train.csv", names=col_names).to_numpy()
    test = pd.read_csv("Data/test.csv", names=col_names).to_numpy()

    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]

    w_i = np.zeros(8)
    lms = LMS()

    lms.train(x_train, y_train, gamma=0.004, initial_w=w_i, mode="stochastic", t=100000)
    plt.plot(range(len(lms.cost_data)), lms.cost_data)
    plt.title("Lms Cost at step t: gamma = 0.004, stochastic")
    plt.xlabel("step t")
    plt.savefig("Experiments/lms_data_b.png")
    print("learned weighs: ", lms.weights)
    print(lms.solve_weights(x_train, y_train))
    x_test_b = lms.append_b(x_test)
    print(lms.cost(lms.weights, x_test_b, y_test))


if __name__ == "__main__":
    main()
