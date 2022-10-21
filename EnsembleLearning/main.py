from ada_boost import AdaBoost
from ada_boost import prediction_error
import pandas as pd


def main():
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
    f = open("ada_exp.csv", "w")
    f.write("t, train, test \n")
    for t in range(1, 501):
        print(t)
        ensemble = AdaBoost(train, t)

        train_pred = ensemble.predict(x_train)
        test_pred = ensemble.predict(x_test)

        f.write(str(t) + ", " + str(prediction_error(y_train, train_pred)) + ", " + str(prediction_error(y_test, test_pred)) + "\n")




if __name__ == "__main__":
    main()
