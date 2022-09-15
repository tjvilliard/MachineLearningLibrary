import numpy as np
import pandas as pd
import DecisionTree as dt


def main():
    t_1 = pd.DataFrame([
        ["x1", "x2", "x3", "x4", "y"],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0]
    ])



    tree = dt.DecisionTree()

    data = t_1.iloc[1:].to_numpy()
    split = dt.split_data(data, 2).values()
    print(tree.info_gain(data, split))


if __name__ == "__main__":
    main()
