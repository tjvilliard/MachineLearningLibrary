import numpy as np
import pandas as pd
from perceptron import Perceptron


def main():
    col_names = ["variance", "skewness","curtosis","entropy","label"]

    train = pd.read_csv("Data/train.csv", names=col_names)
    train["label"] = train["label"].replace([0], -1)
    train = train.to_numpy()

    test = pd.read_csv("Data/test.csv", names=col_names)
    test["label"] = test["label"].replace([0], -1)
    test = test.to_numpy()

    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]


    # set the learning rate
    r = 0.25
    max_epochs = 10

    # standard perceptron
    error_rates = []
    for i in range(100):
        s_perceptron = Perceptron(gamma=r, mode="standard")
        s_perceptron.train(x_train, y_train, t=max_epochs)
        s_perceptron_predictions = s_perceptron.predict(x_test)
        error_rates.append(prediction_error(y_test, s_perceptron_predictions))

    print("Standard Perceptron:")
    print("w of final perceptron = ", s_perceptron.w)
    avg_err = sum(error_rates) / len(error_rates)
    print("Avg Error for 100 Standard Perceptrons: ", avg_err)
    print("\n")

    # voted perceptron
    v_perceptron = Perceptron(gamma=r, mode="voted")
    v_perceptron.train(x_train, y_train, t=max_epochs)
    print("Voted Perceptron:")
    print("[w_i, c_i] = ", v_perceptron.w[:2], "...", v_perceptron.w[-1])
    v_perceptron_predictions = v_perceptron.predict(x_test)
    print("Error: ", prediction_error(y_test, v_perceptron_predictions))
    print("\n")

    # voted perceptron
    a_perceptron = Perceptron(gamma=r, mode="average")
    a_perceptron.train(x_train, y_train, t=max_epochs)
    print("Average Perceptron:")
    print("w = ", a_perceptron.w)
    a_perceptron_predictions = a_perceptron.predict(x_test)
    print("Error: ", prediction_error(y_test, a_perceptron_predictions))


def prediction_error(actual, prediction):
    if len(actual) != len(prediction):
        print("arrays not equal length")
        return None
    error = np.equal(actual, prediction)
    error_rate = 1 - (sum(error) / len(actual))
    return error_rate









if __name__ == "__main__":
    main()