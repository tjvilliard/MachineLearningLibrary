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


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


class AdaBoost:

    def __init__(self, data, num_trees=10):
        # boosting algorithm for decision stumps
        # num_trees gives the number of stumps created in the ensemble
        self.num_trees = num_trees

        # all stumps in ensemble
        self.stumps = []  # empty array for holding stumps
        self.amount_say = []

        self.labels = data[:, -1]
        self.data = data
        #print("Begin boost \n")
        self.tree_boost()

    def predict(self, new_data):
        # only works for +1 -1 classifications
        t = len(self.stumps)
        final_votes = np.zeros(len(new_data))

        # for every stump predict data and scale by amount of say
        # add all vote vectors and sign of each element is prediction
        for i in range(t):
            stump = self.stumps[i]
            votes = stump.predict(new_data) * self.amount_say[i]
            final_votes += votes

        #print("stump predictions done")
        sign_vec = np.vectorize(sign)
        prediction_arr = sign_vec(final_votes)

        return np.array(prediction_arr)

    def tree_boost(self):  ##############################################################################3
        # initialize weight vector
        m = len(self.data)
        weights = np.array([1 / m] * m)
        train_data = self.data

        # one round for every stump created
        for i in range(self.num_trees):
            #print("building stump " + str(i))
            # create stump
            stump = dt.DecisionTree(max_depth=1, data=train_data, handle_numeric=True)

            amount_say, predictions = self.amount_of_say(weights, stump)

            # update weight vector
            weights = self.update_weight_vec(weights, amount_say, predictions)

            # weighted sampling of new training data
            weighted_data_select = choice(a=range(m), size=m, p=weights)
            train_data = self.data[weighted_data_select]  # self.get_selected_data(weighted_data_select)

            # track round data
            self.stumps.append(stump)
            self.amount_say.append(amount_say)

    def get_selected_data(self, selected_idxs):
        """ given an array of indeces, returns new set with datapoints from og set with that idx
        """
        new_data = []

        for index in selected_idxs:
            new_data.append(self.data[index])

        return np.array(new_data)

    def update_weight_vec(self, weights, amount_say, predictions):
        """update the weight vector depending on which datapoints are classified correctly
        weights: numpy array"""
        y = self.labels.astype(float)
        adjustment = np.exp(predictions * y * amount_say * -1)
        weights = weights * adjustment

        # normalize the weight vector
        l1_norm = sum(weights)
        return weights / l1_norm

    def amount_of_say(self, weights, tree: dt.DecisionTree):
        """ determines the amount of say for a given stump"""

        total_error, predictions = self.total_error(weights, tree)

        amount_say = .5 * np.log((1 - total_error) / total_error)

        return amount_say, predictions

    def total_error(self, weights, tree: dt.DecisionTree, ):
        predictions = tree.predict(self.data[:, :-1]).astype(float)

        y = self.labels.astype(float)
        err = predictions * y * weights
        et = 0.5 - (.05 * sum(err))
        return et, predictions
