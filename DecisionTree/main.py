import numpy as np
import pandas as pd
import decision_tree as dt


def main():
    # assignment_part_1()

    assignment_part_2_car()


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
    ], columns=["O", "T", "H", "W", "Play"])


def assignment_part_2_car():
    car_train_data = pd.read_csv("data\\ml_car\\train.csv",
                                 names=["buying","maint", "doors", "persons", "lug_boot", "safety", "label"])
    car_test_data = pd.read_csv("data\\ml_car\\test.csv",
                                names=["buying","maint", "doors", "persons", "lug_boot", "safety", "label"])
    print(car_train_data.head())


def assignment_part_2_bank():
    return


if __name__ == "__main__":
    main()
