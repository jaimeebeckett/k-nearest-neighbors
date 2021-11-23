import collections
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


def radial_basis_fn_gaussian_kernel(self, distance, sigma):
    gaussian_kernel = np.exp(-(distance**2) / (2*sigma ** 2))
    return gaussian_kernel


class KNearestNeighbor:
    def __init__(self, target, k=1, sigma=None):
        self.k = k
        self.target = target
        self.sigma = sigma
        self.predictions = None

    def calculate_knn(self, train, train_label, test):
        distance = euclidean_distance(train, test)
        if self.sigma:
            distance = radial_basis_fn_gaussian_kernel(distance, self.sigma)
        k_neighbors = np.argsort(distance, axis=0)[:self.k]
        self.predict_class(train_label.to_list(), k_neighbors)

    def predict_class(self, label, k_neighbors):
        neighbors = pd.DataFrame(k_neighbors).replace(dict(enumerate(label)))
        if self.sigma:
            self.predictions = neighbors.mean().values
        else:
            self.predictions = neighbors.mode().values

    def cross_validation(self, data, folds, label, evaluation):
        scores = []
        for i in range(5):
            test = data.folds[i]
            train = data.drop(folds[i].index)

            train_label = train[label]
            train_data = train.drop(label, axis=1)
            test_label = test[label].to_numpy()
            test = test.drop(label, axis=1)

            self.calculate_knn(train_data, train_label, test)
            scores.append(evaluation(test_label, self.predictions))
        return scores


class CondensedNearestNeighbor(KNearestNeighbor):
    def __init__(self, train, test, target):
        super().__init__(train, test, target)

    def calculate_knn(self, train, label, test):
        shuffled_df = train.sample(frac=1)
        z = pd.DataFrame()
        for index, row in shuffled_df.iterrows():
            if len(z) < self.k:
                z = z.append(row)
            else:
                distance = euclidean_distance(z.drop(columns=label), row.drop(label))
                if self.sigma:
                    distance = radial_basis_fn_gaussian_kernel(distance, self.sigma)
                neighbor_classes = k_neighbors = np.argsort(distance, axis=0)[:self.k]
                self.predict_class(label.to_list(), k_neighbors)
                if neighbor_classes != row[label]:
                    z = z.append(row)
        return z


class EditedNearestNeighbor(KNearestNeighbor):
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
                    distance = radial_basis_fn_gaussian_kernel(distance, self.sigma)
                k_neighbors = np.argsort(distance, axis=0)[:self.k]
                self.predict_class(label.to_list(), k_neighbors)
                if k_neighbors != row[label]:
                    z = z.drop(index)
        return z
