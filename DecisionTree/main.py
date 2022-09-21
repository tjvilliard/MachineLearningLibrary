import numpy as np
import pandas as pd
import DecisionTree as dt


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



    data = t_1.iloc[:].to_numpy()
    tree = dt.DecisionTree(data)
    for row in data:
        print(tree.predict(row[:-1]))
    print(tree.predict([1,1,1,1]))






if __name__ == "__main__":
    main()
