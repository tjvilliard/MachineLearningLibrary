from ada_boost import AdaBoost
from ada_boost import prediction_error
from bagged_trees import BaggedTrees
import pandas as pd
import numpy as np
from numpy.random import choice


def main():
    # set up
    col_names = range(17)
    bank_train_df = pd.read_csv("data\\ml_bank\\train.csv",
                                names=col_names)

    bank_test_df = pd.read_csv("data\\ml_bank\\test.csv",
                               names=col_names)

    # my implementation requires -1, 1 binary labels
    bank_train_df[16] = bank_train_df[16].replace({"yes": 1, "no": -1})
    bank_test_df[16] = bank_test_df[16].replace({"yes": 1, "no": -1})

    train = bank_train_df.to_numpy()
    test = bank_test_df.to_numpy()
    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]

    # Begin assignment operations
    try:
        assignment_2a(train, x_train, y_train, x_test, y_test)
    except:
        print("something went wrong 2a")

    try:
        assignment_2b(train, x_train, y_train, x_test, y_test)
    except:
        print("something went wrong 2b")

    try:
        assignment_2c(train, x_train, y_train, x_test, y_test)
    except:
        print("something went wrong 2c")

    try:
        assignment_2d(train, x_train, y_train, x_test, y_test)
    except:
        print("something went wrong 2d")


def assignment_2a(train, x_train, y_train, x_test, y_test):
    print("Logging AdaBoost Experiment Data...")
    exp = open("Experiments/ada_exp.csv", "w")

    exp.write("t,train,test\n")
    stump_tr_errors = []
    stump_test_errors = []
    # begin experiment and track data to be logged in csv's
    ensemble = AdaBoost(train, 500)
    for t in range(500):
        print(t)
        train_pred = ensemble.predict(x_train, ensemble_size=t)
        for stump_pred in ensemble.votes:
            stump_tr_errors.append(prediction_error(y_train, stump_pred))

        test_pred = ensemble.predict(x_test, ensemble_size=t)
        for stump_pred in ensemble.votes:
            stump_test_errors.append(prediction_error(y_test, stump_pred))

        exp.write(str(t) + ", " + str(prediction_error(y_train, train_pred)) + ", " + str(
            prediction_error(y_test, test_pred)) + "\n")
    exp.close()
    print("Complete: Experiments/ada_exp.csv\n")

    print("Logging Stump Errors... ")
    st_data = open("Experiments/stump_data.csv", "w")
    st_data.write("train, test \n")
    for i in range(len(stump_tr_errors)):
        st_data.write(str(np.round(stump_tr_errors[i])) + ", " + str(np.round(stump_test_errors[i], 3)) + "\n")
    st_data.close()
    print("Complete: Experiments/stump_data.csv \n")


def assignment_2b(train, x_train, y_train, x_test, y_test):

    print("Logging Bagged Trees Data..")

    f = open("Experiments/bagged_trees_data.csv", "w")
    f.write("t, train, test \n")

    ensemble = BaggedTrees(train, len(train), 500)
    for t in range(500):
        print(t)

        train_pred = ensemble.predict(x_train, t)
        test_pred = ensemble.predict(x_test, t)
        f.write(str(t) + "," + str(prediction_error(y_train, train_pred)) + "," + str(
            prediction_error(y_test, test_pred)) + "\n")

    f.close()
    print("Complete: Experiments/bagged_trees_data.csv \n")


def assignment_2c(train, x_train, y_train, x_test, y_test):

    print("Begin 2c")
    single_trees = []
    bagged_trees = []
    for i in range(100):
        # train the ensemble on a random subset of size 1000
        data = train[choice(range(len(train)), 1000, replace=False)]

        # the bagsize is equal to the length of the training data: |m'| = |m|
        ensemble = BaggedTrees(data, len(data), 500)

        # get a set of 100 ensembles and 100 single trees
        bagged_trees.append(ensemble)
        single_trees.append(ensemble.trees[0])

    single_tree_train_predictions = []
    single_tree_test_predictions = []
    bagged_tree_train_predictions = []
    bagged_tree_test_predictions = []
    for i in range(100):
        single_tree_test_predictions.append(single_trees[i].predict(x_test))

        bagged_tree_test_predictions.append(bagged_trees[i].predict(x_test, 500))

    pd.DataFrame(single_tree_train_predictions).to_csv("Experiments/single_tree_train_prediction.csv")
    pd.DataFrame(single_tree_test_predictions).to_csv("Experiments/single_tree_test_prediction.csv")

    pd.DataFrame(bagged_tree_train_predictions).to_csv("Experiments/bagged_tree_train_prediction.csv")
    pd.DataFrame(bagged_tree_test_predictions).to_csv("Experiments/bagged_tree_test_prediction.csv")

    return


def assignment_2d(train, x_train, y_train, x_test, y_test):
    print("Begin 2d")
    rf_2_train_err = []
    rf_2_test_err = []

    rf_4_train_err = []
    rf_4_test_err = []

    rf_6_train_err = []
    rf_6_test_err = []

    rf_2 = BaggedTrees(train, len(train), 500, random=True, num_attr=2)
    rf_4 = BaggedTrees(train, len(train), 500, random=True, num_attr=4)
    rf_6 = BaggedTrees(train, len(train), 500, random=True, num_attr=6)
    for i in range(100,600, 500):
        print(i)

        rf_2_train_err.append(prediction_error(y_train, rf_2.predict(x_train, i)))
        rf_2_test_err.append(prediction_error(y_test, rf_2.predict(x_test, i)))

        rf_4_train_err.append(prediction_error(y_train, rf_4.predict(x_train, i)))
        rf_4_test_err.append(prediction_error(y_test, rf_4.predict(x_test, i)))

        rf_6_train_err.append(prediction_error(y_train, rf_6.predict(x_train, i)))
        rf_6_test_err.append(prediction_error(y_test, rf_6.predict(x_test, i)))
        print("predictions complete")

    print("Logging Random Forrest Data...")
    file1 = open("Experiments/rf_data_2.csv", "w")
    file1.write("t, train_err, test_err \n")

    file2 = open("Experiments/rf_data_4.csv", "w")
    file2.write("t, train_err, test_err \n")

    file3 = open("Experiments/rf_data_6.csv", "w")
    file3.write("t, train_err, test_err \n")

    for t in range(len(rf_2_train_err)):
        file1.write(str(t) + "," + str(rf_2_train_err[t]) + "," + str(rf_2_test_err[t]) + "\n")
        file2.write(str(t) + "," + str(rf_4_train_err[t]) + "," + str(rf_4_test_err[t]) + "\n")
        file3.write(str(t) + "," + str(rf_6_train_err[t]) + "," + str(rf_6_test_err[t]) + "\n")

    print("Complete: Experiment/rf_data_2.csv, Experiment/rf_data_4.csv, Experiment/rf_data_6.csv")
    return

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


if __name__ == "__main__":
    main()
