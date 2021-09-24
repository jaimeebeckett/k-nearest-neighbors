import collections
import time
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from math import sqrt, floor


def k_fold(df, label, k=5, categorical=True, validation=False):
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


def k_nearest_neighbors(train, test, label, k=1):
    predicted = []

    train_no_labels = train.drop(columns=label)
    test_no_labels = test.drop(columns=label)
    for i in range(len(test)):
        distance = euclidean_distance(train_no_labels, test_no_labels.iloc[i])
        k_neighbors = distance.nsmallest(k)
        neighbor_classes = train.loc[k_neighbors.index]
        predicted.append(find_class(neighbor_classes, label))

    #distance = test_no_labels.apply(lambda x: euclidean_distance(x, train_no_labels), axis=1)
    #k_neighbor_classes = {}
    #for col in distance.T:
    #    classes = train['class'].loc[distance.T[col].nsmallest(k).index].to_list()
    #    most_frequent_class = collections.Counter(classes).most_common()[0][0]
    #    k_neighbor_classes[col] = most_frequent_class
    return predicted


def find_class(df, label):
    classes = collections.Counter(df[label])
    return classes.most_common(1)[0][0]


def condensed_nearest_neighbor(train, test, label):
    z = []
    #for each datapoint in train
    # find x in Z such that ||x - x'|| = min(x in z)||x - xj||
    #if class x != class x', add x to z

def euclidean_distance(train, test):
    distance = (train.subtract(test)).abs().sum(axis=1)
    return distance

