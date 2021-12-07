from collections import Counter
from data_processing import load_car, load_cancer, load_vote, load_machine, load_abalone, \
    load_forest_fire, split_train_test
import numpy as np
from abc import ABC, abstractmethod
from evaluation import classification, mse, split_train_test
import time


def timer(func):
    def wrapper(*args):
        before = time.time()
        x = func(*args)
        print(f'function took {time.time() - before} seconds')
        return x

    return wrapper


class NearestNeighbor(ABC):
    def __init__(self, label, k=1, z=None):
        self.k = k
        self.target = label
        self.predictions = []
        self.scores = []

        assert z is None or z == 'edited' or z == 'condensed'
        self.z = z

    @timer
    def predict(self, train, train_label, test):
        """
        Predict the test labels
        :param train: the training dataset
        :param train_label: the labels for the training dataset
        :param test: the test dataset
        :return: a list of the predicted labels
        """
        predictions = []
        for row in test:
            distance = self.euclidean_distance(train, row)
            neighbors = self.find_nearest_neighbors(distance, train_label)
            prediction = self.make_prediction(neighbors)
            predictions.append(prediction)
        return predictions

    def find_nearest_neighbors(self, distances, labels):
        """
        Calculates the nearest neighbors
        :param distances: A numpy 2D array containing distances
        :param labels: the labels of each neighbor
        :return: the k nearest neighbors and their labels
        """
        args = np.argsort(distances)[:self.k]
        return distances[args], labels[args]

    def cross_validation(self, d, folds, evaluation):
        """
        Find each score with cross validation
        :param d: The dataset
        :param folds: The folds for the dataset
        :param evaluation: the evaluation method
        :return: A list of scores
        """
        self.scores = []
        for fold_num in range(len(folds)):
            train, train_label, test, test_label = split_train_test(d, folds[fold_num], self.target)
            if self.z:
                train, train_label = self.calculate_z(train, train_label)
            predictions = self.predict(train, train_label, test)
            self.scores.append(evaluation(test_label, predictions))
        return self.scores

    def condense_z(self, z, z_labels, test, test_label):
        keep_searching = False
        test_keep = []
        for index, row in enumerate(zip(test, test_label)):
            distance = self.euclidean_distance(z, row[0])
            neighbors = self.find_nearest_neighbors(distance, np.array(z_labels))
            prediction = self.make_prediction(neighbors)
            if prediction != row[1]:
                z = np.vstack([z, row[0]])
                z_labels.append(row[1])
                keep_searching = True
            else:
                test_keep.append(index)
        test_label = test_label[test_keep]
        test = test[test_keep]
        if keep_searching:
            self.condense_z(z, z_labels, test, test_label)
        return z, np.array(z_labels)

    def calculate_z(self, train, train_label):
        shuffler = np.random.permutation(len(train_label))
        train = train[shuffler]
        train_label = train_label[shuffler]
        if self.z == 'condensed':
            z = np.array(train[:self.k])
            labels = list(train_label[:self.k])
            test = train[self.k:]
            test_label = train_label[self.k:]
            z, labels = self.condense_z(z, labels, test, test_label)
        else:
            z = np.array(train)
            labels = list(train_label)

        return z, labels

    def __str__(self):
        return f"""
         Nearest Neighbors:
         Best K: {self.k}
         Fold Scores: {self.scores}, Average Score: {np.mean(self.scores)}
         """

    def train(self, train, train_label):
        pass

    @abstractmethod
    def make_prediction(self, k_neighbors):
        pass

    @staticmethod
    def euclidean_distance(train, test):
        return np.sqrt(np.sum((train - test)**2, axis=1))


class KNearestNeighborRegression(NearestNeighbor):
    def __init__(self, label, k=1, sigma=None, epsilon=None):
        super().__init__(label, k)
        self.sigma = sigma
        self.epsilon = epsilon

    def radial_basis_fn_gaussian_kernel(self, distance):
        """
        Radial Basis Gaussian Kernel
        :param distance: The distances of each neighbor
        :return: The adjusted values
        """
        gaussian_kernel = np.exp(-(distance ** 2) / (2 * (self.sigma ** 2)))
        return gaussian_kernel

    def make_prediction(self, k_distance):
        """
        Predict the label of the test point
        :param k_distance: The k nearest neighbors and their distances
        :return: The predicted label with the kernel adjustment
        """
        kernel = self.radial_basis_fn_gaussian_kernel(k_distance[0])
        return np.mean(kernel * k_distance[1])


class KNearestNeighborClassification(NearestNeighbor):
    def make_prediction(self, k_neighbors):
        return Counter(k_neighbors[1]).most_common()[0][0]


if __name__ == '__main__':
    car = load_car()
    cancer = load_cancer()
    vote = load_vote()
    abalone = load_abalone()
    forest = load_forest_fire()
    machine = load_machine()

    print('car KNN')
    car.k_fold(validation=True)
    knn_car = KNearestNeighborClassification(car.label, k=3)
    knn_car.cross_validation(car.df, car.folds, classification)
    print(knn_car)

    print('car CNN')
    car.k_fold(validation=True)
    cnn_car = KNearestNeighborClassification(car.label, k=3, z='condensed')
    cnn_car.cross_validation(car.df, car.folds, classification)
    print(cnn_car)

    print('cancer KNN')
    cancer.k_fold(validation=True)
    cancer_knn = KNearestNeighborClassification(cancer.label, k=3)
    cancer_knn.cross_validation(cancer.df, cancer.folds, classification)
    print(cancer_knn)

    print('cancer CNN')
    cancer.k_fold(validation=True)
    cancer_cnn = KNearestNeighborClassification(cancer.label, k=3, z='condensed')
    cancer_cnn.cross_validation(cancer.df, cancer.folds, classification)
    print(cancer_cnn)

    print('vote KNN')
    vote.k_fold(validation=True)
    vote_knn = KNearestNeighborClassification(vote.label, k=1)
    vote_knn.cross_validation(vote.df, vote.folds, classification)
    print(vote_knn)

    print('vote CNN')
    vote.k_fold(validation=True)
    vote_cnn = KNearestNeighborClassification(vote.label, k=1, z='condensed')
    vote_cnn.cross_validation(vote.df, vote.folds, classification)
    print(vote_cnn)

    print("abalone")
    abalone_knn = KNearestNeighborRegression(abalone.label, k=8, sigma=4)
    abalone.k_fold()
    abalone_knn.cross_validation(abalone.df, abalone.folds, mse)
    print(abalone_knn)

    print("forest")
    forest.k_fold(validation=True)
    forest_knn = KNearestNeighborRegression(forest.label, k=19, sigma=500)
    forest_knn.cross_validation(forest.df, forest.folds, mse)
    print(forest_knn)

    print("machine")
    machine.k_fold(validation=True)
    machine_knn = KNearestNeighborRegression(machine.label, k=1, sigma=10)
    machine_knn.cross_validation(machine.df, machine.folds, mse)
    print(machine_knn)