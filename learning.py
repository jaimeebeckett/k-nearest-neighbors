from collections import Counter
import pandas as pd
import numpy as np
from math import fabs


def euclidean_distance(train, test):
    """
    Finds all of the euclidean distances between a data point and a DataFrame of data points
    :param train: A DataFrame
    :param test: A DataFrame
    :return: A pandas Series of euclidean distances
    """
    broadcast = train.to_numpy()[:, np.newaxis] - test.to_numpy()
    distance = np.abs(np.sum(broadcast**2, axis=2))
    return pd.DataFrame(distance).set_index(train.index)


class NearestNeighbor:
    def __init__(self, label, k=1, sigma=None):
        self.k = k
        self.target = label
        self.sigma = sigma
        self.predictions = []

    def find_nearest_neighbors(self, distances, labels):
        args = np.argsort(distances)[:self.k]
        return distances[args], labels[args]

    def radial_basis_fn_gaussian_kernel(self, distance):
        gaussian_kernel = np.exp(-(distance ** 2) / (self.k * (self.sigma ** 2)))
        return gaussian_kernel

    @staticmethod
    def predict_class(k_neighbors):
        return Counter(k_neighbors).most_common()[0][0]

    @staticmethod
    def predict_regression(k_neighbors):
        return np.mean(k_neighbors)

    def predict(self, train, train_label, test):
        predictions = []
        for row in test:
            distance = self.euclidean_distance(train, row)
            k_distance, k_label = self.find_nearest_neighbors(distance, train_label)
            if self.sigma:
                kernel = self.radial_basis_fn_gaussian_kernel(k_distance)
                neighbor_mean = self.predict_regression(k_distance * kernel)
                predictions.append(neighbor_mean)
            else:
                predictions.append(self.predict_class(k_label))
        return predictions

    @staticmethod
    def euclidean_distance(train, test):
        return np.sqrt(np.sum((train - test)**2, axis=1))

    def cross_validation(self, data, folds, label, evaluation):
        scores = []
        for i in range(len(folds)):
            test = folds[i].drop(label, axis=1).to_numpy()
            test_label = folds[i][label].to_numpy()

            train = data.drop(folds[i].index)
            train_label = train[label].to_numpy()
            train = train.drop(label, axis=1).to_numpy()

            predictions = self.predict(train, train_label, test)
            scores.append(evaluation(test_label, predictions))
        return scores


class CondensedNearestNeighbor(NearestNeighbor):
    def __init__(self, label, k=1, sigma=None):
        super().__init__(label, k, sigma)

    def predict(self, train, train_label, test):
        z, labels = self.calculate_cnn(train, train_label)
        predictions = super().predict(z, labels, test)
        return predictions

    def calculate_cnn(self, train, train_label):
        shuffler = np.random.permutation(len(train_label))
        train = train[shuffler]
        train_label = train_label[shuffler]
        z = np.array(train[:self.k])
        labels = list(train_label[:self.k])

        for row, label in zip(train[self.k:], train_label[self.k:]):
            distance = self.euclidean_distance(z, row)
            if self.sigma:
                distance = self.radial_basis_fn_gaussian_kernel(distance)
            k_neighbors = np.sort(np.array([distance, labels]))[1, :self.k]
            predicted_label = self.predict_class(k_neighbors)
            if predicted_label != label:
                z = np.vstack([z, row])
                labels.append(label)
        return z, np.array(labels)

    def euclidean_distance(self, train, test):
        return np.sqrt(np.sum((train - test)**2, axis=1))


class EditedNearestNeighbor(NearestNeighbor):
    def __init__(self, train, test, target):
        super().__init__(train, test, target)

    def calculate_knn(self, train, label, test):
        shuffled_df = train.sample(frac=1)
        z = pd.DataFrame(train)
        for index, row in shuffled_df.iterrows():
            if len(z) < self.k:
                z = z.append(row)
            else:
                distance = euclidean_distance(z.drop(index).drop(columns=label), row.drop(label))
                if self.sigma:
                    distance = self.radial_basis_fn_gaussian_kernel(distance)
                k_neighbors = np.argsort(distance, axis=0)[:self.k]
                self.predict_class(label.to_list(), k_neighbors)
                if k_neighbors != row[label]:
                    z = z.drop(index)
        return z
