import collections
import time
from timeit import default_timer as timer

import pandas as pd
from math import sqrt


def k_fold(df, label, k=5, categorical=True, validation=False):
    folds = {}
    valid_data = pd.DataFrame()
    valid_fold_size = 0
    if categorical:  # if we need to stratify y
        df = df.sort_values(label)
    for i in range(k):
        if validation:
            valid_fold_size = (len(df) * 0.20)//k
            valid_data = valid_data.append(df[df.index % k == i][:valid_fold_size - 1])
        folds[i] = df[df.index % k == i][valid_fold_size:]
    return folds, valid_data


def k_nearest_neighbors(train, test, label, k=1):
    predicted = []
    for i, row in test.iterrows():
        distance = euclidean_distance(train.drop(columns=label), row.drop(columns=label))
        k_neighbors = distance.nsmallest(k)
        neighbor_classes = train.loc[k_neighbors.index]
        predicted.append(find_class(neighbor_classes, label))
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
