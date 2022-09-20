import numpy as np


def entropy(y):
    # get unique labels
    labels = np.unique(y[:])

    # sum over entropy given proportion of label to data
    ent = 0
    for l in labels:
        plabel = len(y[y == l]) / len(y)
        ent += -plabel * np.log2(plabel)
    return ent

def majority(y):
    # labels
    labels = np.unique(y)


def split_data(x, col_number):
    splits = {}
    vals = np.unique(x[:, col_number])
    for val in vals:
        subset = np.array([row for row in x if row[col_number] == val])
        # dont track empty subsets
        if len(subset) > 0:
            splits[val] = subset
    return splits


class Node:
    def __init__(self, data=None, split_idx=0, children=None, info_gain=-1, is_leaf=False):
        self.data = data
        self.split_idx = split_idx
        self.children = children
        self.info_gain = info_gain
        self.is_leaf = is_leaf


class DecisionTree:
    def __init__(self, data, min_samples=2, max_depth=2):
        self.min_samples = min_samples
        self.max_depth = max_depth

        self.root = self.build_tree(data)

    def build_tree(self, data, depth=0):
        # check the shape of the data to get the size of data
        x, y = data[:, :-1], data[:, -1]
        num_samples, num_attrs = x.shape()

        if depth <= self.max_depth or num_samples > self.min_samples:
            split_node = self.find_best_split(data)

            if split_node.info_gain > 0:
                # if split_node has no children
                if len(split_node.children) == 0:
                    split_node.is_leaf = True
                    return split_node
                #
                for key in split_node.children.keys():
                    # replace the child data set with child tree
                    split_node.children[key] = self.build_tree(split_node.children[key], depth+1)

        # if stopping conditions met, continues here
        return Node(data=data, is_leaf=True)

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
