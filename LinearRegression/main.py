import pandas as pd
import numpy as np
from lms import LMS

def main():
    w = np.array([0,0,0,0])
    w1 = np.array([-1, -1, 1, -1])
    y = np.array([1, 4, -1, -2, 0])
    x = np.array([[1, -1, 2],
                  [1, 1, 3],
                  [-1, 1, 0],
                  [1, 2, -4],
                  [3, -1, 1]])

    lms= LMS()
    lms.train(x, y, initial_w=w, gamma=0.1, mode="stochastic")
    print(lms.weights)
    print("\n")
    lms.train(x, y, initial_w=w1, gamma=0.1, mode="stochastic")
    print(lms.weights)
    w_star = lms.solve_weights(x, y)
    bias_column = np.ones(shape=(len(x), 1))
    x_b = np.concatenate((bias_column, x), axis=1)
    min_cost = lms.cost(w_star, x_b, y)
    print(w_star, " ", min_cost)


    #assignment()

def assignment():
    # concrete data
    col_names = range(8)
    train = pd.read_csv("Data/train.csv", names=col_names).to_numpy()
    test = pd.read_csv("Data/test.csv", names=col_names).to_numpy()

    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]

    w_i = np.zeros(8)
    lms = LMS()

    for gamma in [1, .5, .25, .125, .0625, .03125, .0150625]:
        print(gamma)
        lms.train(x_train, y_train, gamma, initial_w=w_i, mode="stochastic")
        print("\n")

    print(lms.weights)
    print(lms.solve_weights(x_train, y_train))












if __name__ == "__main__":
    main()