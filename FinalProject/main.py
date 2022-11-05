import pandas as pd
import numpy as np
from DecisionTree.decision_tree import DecisionTree
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import cProfile



def main():
    # import the data
    income_train = pd.read_csv("Data/train_final.csv")
    # income_test is the submission test set that has no y column
    income_test = pd.read_csv("Data/test_final.csv")
    income_test_x = income_test.to_numpy()[:, :-1]

    # generate a train-test split
    income_train = income_train.to_numpy()
    x, y = income_train[:, :-1], income_train[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)


    clf = svm.SVC()
    clf.fit(x_train, y_train)
    clf.predict(x_test)



def decision_tree_classify():

    # import the data
    income_train = pd.read_csv("Data/train_final.csv")
    # income_test is the submission test set that has no y column
    income_test = pd.read_csv("Data/test_final.csv")
    income_test_x = income_test.to_numpy()[:, :-1]

    income_train = income_train.to_numpy()
    x, y = income_train[:, :-1], income_train[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

    train_data = np.concatenate((x_train, y_train.reshape((-1, 1))), axis=1)
    # full tree
    max_depth = income_train.shape[1] - 1

    # Decision Tree first
    income_tree = DecisionTree(data=train_data, mode="entropy", max_depth=max_depth, handle_numeric=True)
    y_test_predictions = income_tree.predict(x_test)

    print(prediction_error(y_test, y_test_predictions))

    sub_tree = DecisionTree(data=income_train, mode="entropy", max_depth=max_depth, handle_numeric=True)
    sub_predict = sub_tree.predict(income_test_x)
    f = open("Data/submission.csv", "w")
    f.write("ID,Prediction\n")
    for i in range(len(sub_predict)):
        f.write(str(i) + "," + str(sub_predict[i]) + "\n")

    f.close()


def prediction_error(actual, prediction):
    if len(actual) != len(prediction):
        print("arrays not equal length")
        return None
    error = np.equal(actual, prediction)
    error_rate = 1 - (sum(error) / len(actual))
    return error_rate

if __name__ == "__main__":
    main()