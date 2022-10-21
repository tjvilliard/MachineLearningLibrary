import numpy as np
from DecisionTree.decision_tree import DecisionTree
from numpy.random import choice


class BaggedTrees:

    def __init__(self, data, bag_size, num_trees):
        self.trees = []

        self.assemble_trees(data, bag_size, num_trees)

    def assemble_trees(self, data, bag_size, num_trees):
        num_col = np.shape(data)[1] - 1
        for i in range(num_trees):
            batch = choice(data, bag_size, replace=True)
            tree = DecisionTree(batch, max_depth=num_col)
            self.trees.append(tree)

    def predict(self, new_data):
        # 2d array of yhat for all trees
        all_predictions = []
        #final predictions after vote
        true_predictions = []

        for tree in self.trees:
            # append yhat for each tree to predictions: each row is single tree prediction
            all_predictions.append(tree.predict(new_data))

        # every entry in transpose(predictions) is set of all predictions for one datapoint
        # take max and append to final pred
        for col in np.transpose(np.array(all_predictions)):
            vote = max(col, key=col.count)
            true_predictions.append(vote)

        return true_predictions
