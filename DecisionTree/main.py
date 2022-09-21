import numpy as np
import pandas as pd
import decision_tree as dt


def main():
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
    ], columns=["O","T","H","W","Play"])


    data1 = t_1.to_numpy()
    tree = dt.DecisionTree(data1)

    data2 = tennis.to_numpy()
    tree2 = dt.DecisionTree(data2)

    x = 4


if __name__ == "__main__":
    main()
