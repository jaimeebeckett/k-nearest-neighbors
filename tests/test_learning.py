import pandas as pd
from learning import euclidean_distance, k_nearest_neighbors, k_fold, condensed_nearest_neighbor


def test_euclidean_distance():
    # df1 = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
    df1 = pd.DataFrame({'x': [1, 1, 3, 4, 5],
                        'y': [4, 4, 1, 9, 3]})
    # df4 = pd.DataFrame({'x': [1, 7, 3, 2, 2],
    #                     'y': [6, 1, 3, 1, 3]})
    df2 = pd.DataFrame({'x': [1],
                        'y': [6]})
    df3 = pd.DataFrame({'x': [1],
                        'y': [4]})
    # distance_one = euclidean_distance(df1, df2)
    # distance_one_reverse = euclidean_distance(df2, df1)
    # distance_two = euclidean_distance(df3, df4)
    print(type(df3))

    distance = euclidean_distance(df1, df2)
    series = euclidean_distance(df2, df3)

    # assert distance_one.equals(pd.Series([5, 4, 0, 3, 2]))
    # assert distance_one_reverse.equals(pd.Series([5, 4, 0, 3, 2]))
    # assert distance_two.equals(pd.Series([2, 9, 2, 10, 3]))
    assert distance.equals(pd.Series([2, 2, 7, 6, 7]))
    assert series.equals(pd.Series([2]))

def test_k_nearest_neighbor():
    train_class = pd.DataFrame({'x': [1, 1, 3, 4, 5],
                                'y': [4, 5, 1, 9, 3],
                                'class': ['no', 'yes', 'no', 'no', 'yes']})
    test_class = pd.DataFrame({'x': [1, 7, 3, 2],
                               'y': [6, 1, 3, 1],
                               'class': ['yes', 'yes', 'yes', 'no']})
    train_regression = pd.DataFrame({'x': [1, 1, 3, 4, 5],
                                     'y': [4, 5, 1, 9, 3],
                                     'class': [0, 1, 2, 5, 2]})
    test_regression = pd.DataFrame({'x': [1, 7, 3, 2],
                                    'y': [6, 1, 3, 1],
                                    'class': [6, 2, 1, 4]})

    knn_class = k_nearest_neighbors(train_class, test_class, 'class', k=3)
    knn_regression = k_nearest_neighbors(train_regression, test_regression, 'class', classification=False, k=3)

    assert knn_class == ['no', 'no', 'no', 'no']
    assert knn_regression == [2, 4/3, 4/3, 1]

def test_condensed_nearest_neighbor():
    train_class = pd.DataFrame({'x': [1, 1, 3, 4, 5],
                                'y': [4, 5, 1, 9, 3],
                                'class': ['no', 'yes', 'no', 'no', 'yes']})
    test_class = pd.DataFrame({'x': [1, 7, 3, 2],
                               'y': [6, 1, 3, 1],
                               'class': ['yes', 'yes', 'yes', 'no']})
    train_regression = pd.DataFrame({'x': [1, 1, 3, 4, 5],
                                     'y': [4, 5, 1, 9, 3],
                                     'class': [0, 1, 2, 5, 2]})
    test_regression = pd.DataFrame({'x': [1, 7, 3, 2],
                                    'y': [6, 1, 3, 1],
                                    'class': [6, 2, 1, 4]})
    cnn_class = condensed_nearest_neighbor(train_class, test_class, 'class', 3)
    print(cnn_class)

def test_k_fold():
    df = pd.DataFrame({'x': ['x1', 'x2', 'x3', 'x4', 'x5'],
                       'y': [1, 2, 3, 4, 5]})
    folds, valid = k_fold(df, 'y', categorical=False)

    test_folds = {
        0: pd.DataFrame({'x': ['x1'], 'y': [1]}),
        1: pd.DataFrame({'x': ['x2'], 'y': [2]}, index=[1]),
        2: pd.DataFrame({'x': ['x3'], 'y': [3]}, index=[2]),
        3: pd.DataFrame({'x': ['x4'], 'y': [4]}, index=[3]),
        4: pd.DataFrame({'x': ['x5'], 'y': [5]}, index=[4])
    }
    assert len(valid) == 0
    assert folds[0].equals(test_folds[0])
    assert folds[1].equals(test_folds[1])
    assert folds[2].equals(test_folds[2])
    assert folds[3].equals(test_folds[3])
    assert folds[4].equals(test_folds[4])
