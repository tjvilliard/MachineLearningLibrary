import numpy as np
import pandas as pd
from svm import SVM


def main():
    col_names = ["variance", "skewness", "curtosis", "entropy", "label"]

    train = pd.read_csv("Data/train.csv", names=col_names)
    train["label"] = train["label"].replace([0], -1)
    train = train.to_numpy()

    test = pd.read_csv("Data/test.csv", names=col_names)
    test["label"] = test["label"].replace([0], -1)
    test = test.to_numpy()

    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]

    # starting variables
    gamma_0 = .005
    a = .5
    C = [100 / 873, 500 / 873, 700 / 873]
    max_epochs = 100

    #assignment_primal(x_train, y_train, x_test, y_test, max_epochs, a, gamma_0, C)

    assignment_dual(x_train, y_train, x_test, y_test, C)

    gamma = [.1, .5, 1, 5, 100]
    #assignment_kernel(x_train, y_train, x_test, y_test, gamma, C)


def assignment_primal(x_train, y_train, x_test, y_test, max_epochs, a, gamma_0, C):
    print("PRIMAL SOLUTION---------------------------------")
    for c in C:
        # primal solution problems
        svm_schedule1 = SVM(max_epoch=max_epochs, c=c, a=a, gamma=gamma_0, gamma_schedule=1)
        svm_schedule1.train(x_train, y_train)
        schedule1_train_predictions = svm_schedule1.predict(x_train)
        schedule1_test_predictions = svm_schedule1.predict(x_test)

        svm_schedule2 = SVM(max_epoch=max_epochs, c=c, a=a, gamma=gamma_0, gamma_schedule=2)
        svm_schedule2.train(x_train, y_train)
        schedule2_train_predictions = svm_schedule2.predict(x_train)
        schedule2_test_predictions = svm_schedule2.predict(x_test)

        print("C = ", c)
        print("Schedule 1")
        print("w* = ", svm_schedule1.w)
        print("Train Error: ", prediction_error(y_train, schedule1_train_predictions))
        print("Test Error: ", prediction_error(y_test, schedule1_test_predictions))
        print("")
        print("Schedule 2")
        print("w* = ", svm_schedule2.w)
        print("Train Error: ", prediction_error(y_train, schedule2_train_predictions))
        print("Test Error: ", prediction_error(y_test, schedule2_test_predictions))
        print("------------")


def assignment_dual(x_train, y_train, x_test, y_test, C):
    print("DUAL SOLUTION----------------")
    for c in C:
        # dual solution problems

        svm_dual = SVM(c=c, mode="dual")
        svm_dual.train(x_train, y_train)
        svm_dual_train_predictions = svm_dual.predict(x_train)
        svm_dual_test_predictions = svm_dual.predict(x_test)

        print("w* = ", svm_dual.w)
        print("Train Error: ", prediction_error(y_train, svm_dual_train_predictions))
        print("Test Error: ", prediction_error(y_test, svm_dual_test_predictions))

        print("--------------- \n")


def assignment_kernel(x_train, y_train, x_test, y_test, gamma, C):
    for c in C:
        print("C = ", c)
        prev_support_vec = []
        for g in gamma:
            print("gamma = ", g)
            svm_gauss = SVM(c=c, mode="gaussian-kernel", gamma=g)
            svm_gauss.train(x_train, y_train)
            svm_gauss_train_predictions = svm_gauss.predict(x_train)
            svm_gauss_test_predictions = svm_gauss.predict(x_test)


            print("Train Error: ", prediction_error(y_train, svm_gauss_train_predictions))
            print("Test Error: ", prediction_error(y_test, svm_gauss_test_predictions))
            print("# support vectors: ", len(svm_gauss.support_vector_idxs))
            if g > 0.1:
                vec_in_common = set(svm_gauss.support_vector_idxs).intersection(set(prev_support_vec))
                print("shared sv from prev gamma: \n", vec_in_common)
            print("\n")

            prev_support_vec = svm_gauss.support_vector_idxs


def prediction_error(actual, prediction):
    if len(actual) != len(prediction):
        print("arrays not equal length")
        return None
    error = np.equal(actual, prediction)
    error_rate = np.round(1 - (sum(error) / len(actual)), 4)
    return error_rate


if __name__ == "__main__":
    main()
