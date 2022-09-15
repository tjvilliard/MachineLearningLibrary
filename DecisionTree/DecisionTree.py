import numpy as np


def entropy(y):
    # get unique labels
    labels = np.unique(y)

    # sum over entropy given proportion of label to data
    ent = 0
    for l in labels:
        plabel = len(y[y == l]) / len(y)
        ent += -plabel * np.log2(plabel)
    return ent


def split_data(x, col_number):
    splits = {}
    vals = np.unique(x[:, col_number])
    for val in vals:
        splits[val] = np.array([row for row in x if row[col_number] == val])
    return splits


class Node:
    def __init__(self, data=None, split_idx=0, children=None, info_gain=-1):
        self.data = data
        self.split_idx = split_idx
        self.children = children
        self.info_gain = info_gain


class DecisionTree:
    def __init__(self, min_samples=2, max_depth=2):
        self.root = None

        self.min_samples = min_samples
        self.max_depth = max_depth

    def build_tree(self, data, depth=0):
        return

    def find_best_split(self, data):
        ###### so not sure about this node return procedure
        node = Node(data=data)
        max_gain = float('-inf')

        # split the data for each attribute and determine optimal split
        num_attr = data.shape[1] - 1
        for attr_idx in range(num_attr):
            # split data and compute info gain
            split = split_data(data, attr_idx)
            info_gain = self.info_gain(data, split.values())

            if info_gain > max_gain:
                max_gain = info_gain

                node.info_gain = info_gain
                node.split_idx = attr_idx
                node.children = split

        return node

    def info_gain(self, pre_split, splits, mode="entropy"):
        """Calculates information gain from og data set to splits. pre_split should be ndarray with no column
        information, splits should be list of subsets of pre_split """
        # labels of data and splits
        Y = pre_split[:, -1]
        y = []

        # splits are subsets of pre_split based on value of single attribute
        # weight needed for each split
        num_splits = len(splits)
        weights = []
        for s in splits:
            s_lab = s[:, -1]
            y.append(s_lab)
            weights.append(len(s_lab) / len(Y))

        if mode == 'entropy':
            # entropy pre-split
            h_s = entropy(Y)

            # entropy after split
            h_split = 0
            for i in range(num_splits):
                h_split += weights[i] * entropy(y[i])

            return h_s - h_split
