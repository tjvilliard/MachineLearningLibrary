import numpy as np
import pandas as pd
from decision_tree import DecisionTree
from decision_tree import arr_isnumeric


def main():
    #assignment_part_1()

    assignment_part_2_car()

    assignment_part_2_bank()


def assignment_part_1():
    t_1 = pd.DataFrame([
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0]
    ], columns=["x1", "x2", "x3", "x4", "y"])

    tennis = pd.DataFrame([
        ["S", "H", "H", "W", 0],
        ["S", "H", "H", "S", 0],
        ["O", "H", "H", "W", 1],
        ["R", "M", "H", "W", 1],
        ["R", "C", "N", "W", 1],
        ["R", "C", "N", "S", 0],
        ["O", "C", "N", "S", 1],
        ["S", "M", "H", "W", 0],
        ["S", "C", "N", "W", 1],
        ["R", "M", "N", "W", 1],
        ["S", "M", "N", "S", 1],
        ["O", "M", "H", "S", 1],
        ["O", "H", "N", "W", 1],
        ["R", "M", "H", "S", 0],
        ["O", "M", "N", "W", 1],
        ["O", "M", "N", "W", 1]

    ], columns=["O", "T", "H", "W", "Play"])

    placeholder = ["O", "M", "N", "W", 1]
    fill = tennis["W"].where(tennis["Play"]==1)

    tennisnp = tennis.to_numpy()
    fill_val = max(list(tennisnp.T[0]), key=list(tennisnp.T[0]).count)
    tree = DecisionTree(tennisnp)
    for i in range(4):
        splits = DecisionTree.split_data(tennisnp, i)
        i_gain = tree.info_gain(tennisnp, splits.values())
        print("Col " + str(i) + " i_gain: " + str(np.round(i_gain, 3)))


def assignment_part_2_car():
    car_train_df = pd.read_csv("data\\ml_car\\train.csv",
                               names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"])

    car_test_df = pd.read_csv("data\\ml_car\\test.csv",
                              names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"])

    car_train_array = car_train_df.to_numpy()
    car_test_array = car_test_df.to_numpy()
    x_train, y_train = car_train_array[:, :-1], car_train_array[:, -1]
    x_test, y_test = car_test_array[:, :-1], car_test_array[:, -1]



    # dicts to track mode and depth prediction data and average error of each mode
    mode_data_train = {}
    train_avg_err = {}
    mode_data_test = {}
    test_avg_err = {}

    # information gain modes and max depth for iteration
    modes = ["entropy", "gini", "majority"]
    max_depth = 6

    # for each mode build tree of each depth and track prediction data
    for m in modes:

        # avg error of all depths for each mode
        mode_err_sum_train = 0
        mode_err_sum_test = 0

        for i in range(1, max_depth + 1):
            # build tree at depth i from training data
            car_tree = DecisionTree(car_train_array, max_depth=i, mode=m)

            series_name = m + ": Depth = " + str(i)

            # track predictions in dictionary so it can be converted to pd.series/dataframe
            mode_data_train[series_name] = car_tree.predict(x_train)
            mode_data_test[series_name] = car_tree.predict(x_test)

            mode_err_sum_train += prediction_error(y_train, mode_data_train[series_name])
            mode_err_sum_test += prediction_error(y_test, mode_data_test[series_name])

        train_avg_err[m] = np.round(mode_err_sum_train / max_depth, 3)
        test_avg_err[m] = np.round(mode_err_sum_test / max_depth, 3)

    print("Average Error Training Data")
    print(train_avg_err, "\n")
    print("Average Error Test Data")
    print(test_avg_err)

    prediction_train = pd.DataFrame(mode_data_train)
    prediction_test = pd.DataFrame(mode_data_test)

    # put all the data in their own data frames just in case thats necessary
    train_df = pd.concat([car_train_df, prediction_train], axis=1)
    test_df = pd.concat([car_test_df, prediction_test], axis=1)

    # print("debug")


def assignment_part_2_bank():
    col_names = range(17)
    bank_train_df = pd.read_csv("data\\ml_bank\\train.csv",
                                names=col_names)

    bank_test_df = pd.read_csv("data\\ml_bank\\test.csv",
                               names=col_names)

    replace = True
    if replace:
        for name in col_names:
            # replace "unknown" with most common val in the column

            # first for the training df
            train_column = bank_train_df[name]
            most_common_train = train_column.mode()[0]
            bank_train_df[name] = train_column.replace(to_replace="unknown", value=most_common_train)

            # then for the test df
            test_column = bank_test_df[name]
            most_common_test = test_column.mode()[0]
            bank_test_df[name] = test_column.replace(to_replace="unknown", value=most_common_test)

    bank_train_array = bank_train_df.to_numpy()
    bank_test_array = bank_test_df.to_numpy()

    x_train, y_train = bank_train_array[:, :-1], bank_train_array[:, -1]
    x_test, y_test = bank_test_array[:, :-1], bank_test_array[:, -1]

    # dicts to track mode and depth prediction data and average error of each mode
    mode_data_train = {}
    train_avg_err = {}
    mode_data_test = {}
    test_avg_err = {}

    # information gain modes and max depth for iteration
    modes = ["entropy", "gini", "majority"]
    max_depth = 16

    # for each mode build tree of each depth and track prediction data
    for m in modes:

        # avg error of all depths for each mode
        mode_err_sum_train = 0
        mode_err_sum_test = 0

        for i in range(10, max_depth + 1):
            # build tree at depth i from training data
            bank_tree = DecisionTree(bank_train_array, max_depth=i, mode=m, handle_numeric=True)

            series_name = m + ": Depth = " + str(i)
            #print(series_name)

            # track predictions in dictionary so it can be converted to pd.series/dataframe
            mode_data_train[series_name] = bank_tree.predict(x_train)
            mode_data_test[series_name] = bank_tree.predict(x_test)

            try:
                mode_err_sum_train += prediction_error(y_train, mode_data_train[series_name])
                mode_err_sum_test += prediction_error(y_test, mode_data_test[series_name])
            except:
                print("error")

        train_avg_err[m] = np.round(mode_err_sum_train / max_depth, 3)
        test_avg_err[m] = np.round(mode_err_sum_test / max_depth, 3)

    print("Average Error Training Data")
    print(train_avg_err)
    print("Average Error Test Data")
    print(test_avg_err)

    prediction_train = pd.DataFrame(mode_data_train)
    prediction_test = pd.DataFrame(mode_data_test)

    # put all the data in their own data frames just in case thats necessary
    train_df = pd.concat([bank_train_df, prediction_train], axis=1)
    test_df = pd.concat([bank_test_df, prediction_test], axis=1)


def prediction_error(actual, prediction):
    if len(actual) != len(prediction):
        print("arrays not equal length")
        return None
    error = np.equal(actual, prediction)
    error_rate = 1 - (sum(error) / len(actual))
    return error_rate


if __name__ == "__main__":
    main()
