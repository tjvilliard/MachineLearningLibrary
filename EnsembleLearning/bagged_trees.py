import numpy as np
from DecisionTree.decision_tree import DecisionTree
from numpy.random import choice
from sklearn import tree


class BaggedTrees:

    def __init__(self, data, bag_size, num_trees, random=False, num_attr=2):
        self.trees = []
        # determines whether normal bagged trees or random forrest is run
        self.random = random
        # only used when random ==true
        self.num_attr = num_attr

        self.assemble_trees(data, bag_size, num_trees)

    def assemble_trees(self, data, bag_size, num_trees):
        num_col = np.shape(data)[1] - 1

        # build num_trees and store for prediciton
        for i in range(num_trees):
            selection = choice(range(len(data)), bag_size, replace=True)
            batch = data[selection]
            tree = DecisionTree(batch, max_depth=num_col, random=self.random, num_attr=self.num_attr,
                                handle_numeric=True)
            self.trees.append(tree)

    def predict(self, new_data, ensemble_size):
        # 2d array of yhat for all trees
        all_predictions = []

        # final predictions after vote
        true_predictions = []

        for i in range(ensemble_size):
            # append yhat for each tree to predictions: each row is single tree prediction
            all_predictions.append(self.trees[i].predict(new_data))

        # every entry in transpose(predictions) is set of all predictions for one datapoint
        # take max and append to final pred
        for col in np.transpose(np.array(all_predictions)):
            vote = max(col, key=list(col).count)
            true_predictions.append(vote)

        return true_predictions
