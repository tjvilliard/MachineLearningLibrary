import numpy as np


def entropy(y):
    # get unique labels
    labels = np.unique(y)

    # sum over entropy given proportion of label to data
    ent = 0
    for l in labels:
        plabel = len(y[y == l]) / len(y)
        ent += -plabel*np.log2(plabel)
    return ent


def split_data(x, col_number):
    splits = {}
    vals = np.unique(x[:, col_number])
    for val in vals:
        splits[val] = np.array([row for row in x if row[col_number] == val])
    return splits


class DecisionTree:
    def __init__(self, min_samples=2, max_depth=2):
        self.root = None

        self.min_samples = min_samples
        self.max_depth = max_depth

    def build_tree(self, data, depth=0):
        return

    def info_gain(self, pre_split, splits, mode="entropy"):
        # labels of data and splits
        Y = pre_split[:,-1]
        y = []

        # splits are subsets of pre_split based on value of single attribute
        # weight needed for each split
        num_splits = len(splits)
        weights = []
        for s in splits:
            s_lab = s[:,-1]
            y.append(s_lab)
            weights.append(len(s_lab) / len(Y))

        if mode == 'entropy':
            # entropy pre-split
            h_s = entropy(Y)

            # entropy after split
            h_split = 0
            for i in range(num_splits):
                h_split += weights[i]*entropy(y[i])

            return h_s - h_split
