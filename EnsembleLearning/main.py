from ada_boost import AdaBoost
from ada_boost import prediction_error
import pandas as pd
import numpy as np


def main():
    assignment_2a()

    # assignment_2b()

    # assignment_2c()

    # assignment_2d()


def assignment_2a():
    col_names = range(17)
    bank_train_df = pd.read_csv("data\\ml_bank\\train.csv",
                                names=col_names)

    bank_test_df = pd.read_csv("data\\ml_bank\\test.csv",
                               names=col_names)

    bank_train_df[16] = bank_train_df[16].replace({"yes": 1, "no": -1})
    bank_test_df[16] = bank_test_df[16].replace({"yes": 1, "no": -1})

    train = bank_train_df.to_numpy()
    test = bank_test_df.to_numpy()
    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]

    test_errs = []
    train_errs = []

    print("Logging AdaBoost Experiment Data...")
    exp = open("Experiments/ada_exp.csv", "w")

    exp.write("t, train, test \n")
    stump_tr_errors = []
    stump_test_errors = []
    for t in range(1, 501):
        print(t)
        ensemble = AdaBoost(train, t)

        train_pred = ensemble.predict(x_train)
        for stump_pred in ensemble.votes:
            stump_tr_errors.append(prediction_error(y_train, stump_pred))

        test_pred = ensemble.predict(x_test)
        for stump_pred in ensemble.votes:
            stump_test_errors.append(prediction_error(y_test, stump_pred))

        exp.write(str(t) + ", " + str(prediction_error(y_train, train_pred)) + ", " + str(
            prediction_error(y_test, test_pred)) + "\n")
    exp.close()
    print("Complete: Experiments/ada_exp.csv \n")

    print("Logging Stump Errors... ")
    st_data = open("Experiments/stump_data.csv", "w")
    st_data.write("train, test \n")
    for i in range(len(stump_tr_errors)):
        st_data.write(str(np.round(stump_tr_errors[i])) + ", " + str(np.round(stump_test_errors[i], 3)) + "\n")
    st_data.close()
    print("Complete: Experiments/stump_data.csv \n")


def assignment_2b():
    return


def assignment_2c():
    return


def assignment_2d():
    return


if __name__ == "__main__":
    main()
