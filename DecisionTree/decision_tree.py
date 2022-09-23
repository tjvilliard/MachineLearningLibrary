import numpy as np


class Node:
    def __init__(self, data=None, split_idx=0, children=None, info_gain=-1, is_leaf=False):
        self.data = data
        self.split_idx = split_idx
        self.children = children
        self.info_gain = info_gain
        self.is_leaf = is_leaf

    def get_value(self):
        label_list = list(self.data[:, -1])

        # return the label with the highest count in the dataset
        return max(label_list, key=label_list.count)


class DecisionTree:
    def __init__(self, data, mode="entropy", min_samples=1, max_depth=10):

        # arbitrary stopping conditions, user set
        self.min_samples = min_samples
        self.max_depth = max_depth

        # method of information gain: entropy, gini, or majority
        self.mode = mode

        self.root = self.build_tree(data)

    def build_tree(self, data, depth=0):
        """ recursively builds the tree in place using nodes and subsets of the original dataset"""
        # check number of data points
        num_samples = np.shape(data)[0]

        # checking arbitrary stopping conditions
        if depth <= self.max_depth and num_samples >= self.min_samples:
            # set split_node to node of best split
            split_node = self.find_best_split(data)

            # node has no children or node is sorted
            if len(split_node.children) == 0 or split_node.info_gain == 0:
                split_node.is_leaf = True
                return split_node

            # node has children
            for key in split_node.children.keys():
                # replace the child data set with child tree
                split_node.children[key] = self.build_tree(split_node.children[key], depth + 1)

            return split_node

        # if stopping conditions met, continues here
        return Node(data=data, is_leaf=True)

    def find_best_split(self, data):
        """Initializes node from data set. Sets node properties according to those that best split the node."""
        node = Node(data=data)
        max_gain = -1  # info gain cannot be negative

        # split the data for each attribute and determine optimal split
        num_attr = data.shape[1] - 1
        for attr_idx in range(num_attr):
            # split data and compute info gain
            split = self.split_data(data, attr_idx)
            info_gain = self.info_gain(data, split.values())

            if info_gain > max_gain:
                # reset marker and initialize node properties
                max_gain = info_gain
                node.info_gain = info_gain
                node.split_idx = attr_idx
                node.children = split
        # returns node of best split
        return node

    def info_gain(self, pre_split, splits):
        """Calculates information gain from og data set to splits. pre_split should be ndarray with no column
        information, splits should be list of subsets of pre_split """
        # labels of data and splits
        pre_split_labels = pre_split[:, -1]
        split_labels = []

        # splits are subsets of pre_split based on value of single attribute
        # weight needed for each split
        num_splits = len(splits)
        weights = []
        for s in splits:
            s_lab = s[:, -1]
            split_labels.append(s_lab)
            weights.append(len(s_lab) / len(pre_split_labels))

        if self.mode == 'entropy':
            # entropy pre-split
            h_s = self.entropy(pre_split_labels)

            # entropy after split
            h_sv = 0
            for i in range(num_splits):
                h_sv += weights[i] * self.entropy(split_labels[i])

            return h_s - h_sv

        elif self.mode == "gini":
            # pre split gini
            gini_s = self.gini(pre_split_labels)

            gini_sv = 0
            for i in range(num_splits):
                gini_sv += weights[i] * self.gini(split_labels[i])

            return gini_s - gini_sv

        elif self.mode == "majority":
            # pre split ME
            majority_s = self.majority(pre_split_labels)

            majority_sv = 0
            for i in range(num_splits):
                w = weights[i]
                split_i = split_labels[i]
                me_i = self.majority(split_i)
                majority_sv += weights[i] * self.majority(split_labels[i])

            return majority_s - majority_sv

    def predict(self, x):
        """returns ndarray of predictions for all data points in x"""
        if np.shape(x)[1] != np.shape(self.root.data)[1] - 1:
            print("X does not fit data")
            return None

        predictions = []
        for data_point in x:
            predictions.append(self.traverse_tree(self.root, data_point))
        return np.array(predictions)

    def traverse_tree(self, node, x):
        """traverses the tree checking values of x at each node until leaf node is reached. Returns value of reached
        leaf node. """

        # if leaf return value of leaf
        if node.is_leaf:
            val = node.get_value()
            return val

        attr_idx = node.split_idx
        x_val = x[attr_idx]
        if x_val in node.children.keys():
            next_node = node.children[x_val]
            return self.traverse_tree(next_node, x)
        else:
            # if x_val not in node.children, choose most frequent val for fill in
            count = 0
            most_key = ""
            for key in node.children.keys():
                key_count = len(node.children[key].data)
                if key_count > count:
                    count = key_count
                    most_key = key
            next_node = node.children[most_key]
            return self.traverse_tree(next_node, x)

    @staticmethod
    def entropy(y):
        # get unique labels
        labels = np.unique(y[:])

        # sum over entropy given proportion of label to data
        ent = 0
        for label in labels:
            p_label = len(y[y == label]) / len(y)
            ent += -p_label * np.log2(p_label)
        return ent

    @staticmethod
    def majority(y):
        # return proportion of non-majority label

        label_list = list(y)
        majority_label = max(label_list, key=label_list.count)
        majority_p = label_list.count(majority_label) / len(label_list)
        majority_error = 1 - majority_p
        return majority_error

    @staticmethod
    def gini(y):
        # get unique labels
        labels = np.unique(y[:])

        # sum over entropy given proportion of label to data
        gini_sum = 0
        for label in labels:
            p_label = len(y[y == label]) / len(y)
            gini_sum += np.power(p_label, 2)
        return 1 - gini_sum

    @staticmethod
    def split_data(x, col_number):
        splits = {}
        vals = np.unique(x[:, col_number])
        for val in vals:
            subset = np.array([row for row in x if row[col_number] == val])
            # do not track empty subsets
            if len(subset) > 0:
                splits[val] = subset
        return splits
