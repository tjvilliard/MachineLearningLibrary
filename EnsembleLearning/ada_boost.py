from DecisionTree import decision_tree as dt
import numpy as np
from numpy.random import choice


def prediction_error(actual, prediction):
    if len(actual) != len(prediction):
        print("arrays not equal length")
        return None
    error = np.equal(actual, prediction)
    error_rate = 1 - (sum(error) / len(actual))
    return error_rate


class AdaBoost:

    def __init__(self, data, num_trees=10):
        # boosting algorithm for decision stumps
        # num_trees gives the number of stumps created in the ensemble
        self.num_trees = num_trees

        # all stumps in ensemble
        self.stumps = []  # empty array for holding stumps

        self.labels = data[:, -1]
        self.data = data

        self.tree_boost()

    def tree_boost(self):
        # initialize weight vector
        m = len(self.data)
        weights = np.array([1 / m] * m)

        # one round for every stump created
        for i in range(self.num_trees):
            # establish training set for current stump
            weighted_data_select = choice(range(m), m, weights)
            train_data = self.get_selected_data(weighted_data_select)

            # reset weights for new train_ data
            weights = np.array([1 / m] * m)

            # create stump
            stump = dt.DecisionTree(max_depth=1, data=train_data, handle_numeric=True)
            amount_say, errors = self.amount_of_say(stump)

            # update weight vector
            weights = self.update_weight_vec(weights, amount_say, errors)

            # track round data
            self.stumps.append(stump)


    def get_selected_data(self, selected_idxs):
        """ given an array of indeces, returns new set with datapoints from og set with that idx
        """
        new_data = []

        for index in selected_idxs:
            new_data.append(self.data[index])

        return np.array(new_data)


    def update_weight_vec(self, weights, amount_say, errors):
        """update the weight vector depending on which datapoints are classified correctly
        weights: numpy array"""

        for i in range(len(errors)):
            # errors is a boolean vector
            if errors[i]:
                # datapoint i is correctly classified
                w = weights[i]
                weights[i] = w * np.exp(amount_say)

            else:
                # datapoint i is incorrectly classified
                w = weights[i]
                weights[i] = w * np.exp(-1 * amount_say)

        # normalize the weight vector
        l1_norm = sum(weights)
        return weights / l1_norm

    def amount_of_say(self, tree: dt.DecisionTree):
        """ determines the amount of say for a given stump"""
        labels = self.data[:, -1]
        trimmed_data = self.data[:, :-1]

        predictions = tree.predict(trimmed_data)
        errors = np.equal(predictions, labels)
        total_error = sum(errors)

        amount_say = .5 * np.log2((1 - total_error) / total_error)

        return amount_say, errors
