import collections
import time
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from math import sqrt, floor
from evaluation import eval_metric


def k_fold(df, label, k=5, categorical=True, validation=False):
    """
    :param df: DataFrame to fold
    :param label: name of column containing correct values
    :param k: int, number of folds
    :param categorical: boolean, True if categorical, False if regression
    :param validation: boolean, True if taking out a validation set
    :return: a dictionary of the folds and the validation DataFrame
    """
    folds = {}
    valid_data = pd.DataFrame()
    valid_fold_size = 0
    if categorical:  # if we need to stratify y
        df = df.sort_values(label)
    for i in range(k):
        if validation:
            valid_fold_size = floor(len(df) * 0.20)//k
            valid_data = valid_data.append(df[df.index % k == i][:valid_fold_size - 1])
        folds[i] = df[df.index % k == i][valid_fold_size:]
    return folds, valid_data


def k_nearest_neighbors(train, test, label, classification=True, k=1):
    """
    :param train: The training DataFrame
    :param test: The test DataFrame
    :param label: The name of the column containing the correct values
    :param classification: Boolean indicating if the data is classification or regression
    :param k: The number of neighbors
    :return: A list of the predicted values
    """
    predicted = []
    train_no_labels = train.drop(columns=label)
    test_no_labels = test.drop(columns=label)
    for i in range(len(test)):
        distance = euclidean_distance(train_no_labels, test_no_labels.iloc[i])
        k_neighbors = distance.nsmallest(k)
        neighbor_classes = train.loc[k_neighbors.index]
        if classification:
            predicted.append(find_class(neighbor_classes, label))
        else:
            print(neighbor_classes)
            predicted.append(neighbor_classes[label].mean())
    #distance = test_no_labels.apply(lambda x: euclidean_distance(x, train_no_labels), axis=1)
    #k_neighbor_classes = {}
    #for col in distance.T:
    #    classes = train['class'].loc[distance.T[col].nsmallest(k).index].to_list()
    #    most_frequent_class = collections.Counter(classes).most_common()[0][0]
    #    k_neighbor_classes[col] = most_frequent_class
    return predicted


def find_class(df, label):
    """
    :param df: A pandas DataFrame
    :param label: The column name containing the correct classification
    :return: a string, the most common class
    """
    classes = collections.Counter(df[label])
    return classes.most_common(1)[0][0]


def condensed_nearest_neighbor(train, test, label, k, condensed=True):
    #smallest subset z in instead of x such that errors don't increase
    shuffled_df = train.sample(frac=1)
    if condensed:
        z = pd.DataFrame()
    else:
        z = pd.DataFrame(train)
    count = 0
    first = True
    for index, row in shuffled_df.iterrows():
        if first and condensed:
            z = z.append(row)
            first = False
        else:
            row_no_label = row.drop(label)
            distance = euclidean_distance(z.drop(columns=label), row_no_label)
            if z.loc[distance.nsmallest(1).index, label].values != row[label]:
               z = z.append(row)
               count = 0
            else:
                count += 1
    knn_predicted = k_nearest_neighbors(train, test, label, k)
    return knn_predicted


def nearest_neighbor_cross_validation(df, folds, label, k_values=[1, 2, 3, 4, 5], eval_type="classification"):
    """

    :type k_values: list
    """
    knn_best_k = 1
    knn_best_result = 0
    cnn_best_k = 1
    cnn_best_result = 0
    for fold_number in range(len(folds)):
        print(len(folds))
        train = df.drop(folds[fold_number].index)
        test = folds[fold_number]
        knn_predicted = k_nearest_neighbors(train, test, label, k=k_values[fold_number])
        cnn_predicted = condensed_nearest_neighbor(train, test, label, k_values[fold_number])
        actual = test[label].to_list()
        knn_results = eval_metric(actual, knn_predicted, eval_type)
        cnn_results = eval_metric(actual, cnn_predicted, eval_type)
        if knn_results > knn_best_result:
            knn_best_k = k_values[fold_number]
            knn_best_result = knn_results
        if knn_results > cnn_best_result:
            cnn_best_k = k_values[fold_number]
            cnn_best_result = cnn_results

    print(f"Best knn k-value: {knn_best_k}")
    print(f"Best cnn k-value: {cnn_best_k}")
    return knn_best_k, cnn_best_k


def euclidean_distance(dp1, dp2):
    distance = (dp1.subtract(dp2.values)).abs().sum(axis=1)
    return distance

