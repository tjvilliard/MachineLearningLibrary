from ada_boost import AdaBoost
from ada_boost import prediction_error
import pandas as pd


def main():

    assignment_2a()

    #assignment_2b()

    #assignment_2c()

    #assignment_2d()

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
    exp = open("Experiments/ada_exp.csv", "w")
    st_data = open("Experiments/stump_data.csv", "w")

    exp.write("t, train, test \n")
    st_data.write("")
    for t in range(1, 502, 50):
        print(t)
        ensemble = AdaBoost(train, t)

        train_pred = ensemble.predict(x_train)
        test_pred = ensemble.predict(x_test)

        exp.write(str(t) + ", " + str(prediction_error(y_train, train_pred)) + ", " + str(
            prediction_error(y_test, test_pred)) + "\n")


def assignment_2b():

def assignment_2c():

def assignment_2d():

if __name__ == "__main__":
    main()
