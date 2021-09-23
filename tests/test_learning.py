import pandas as pd
from learning import k_fold, euclidean_distance, k_nearest_neighbors


def test_k_fold():
    df = pd.DataFrame({'x': ['x1', 'x2', 'x3', 'x4', 'x5'],
                       'y': [1, 2, 3, 4, 5]})
    folds, valid = k_fold(df, 'y', categorical=False)

    test_folds = {
        0: pd.DataFrame({'x': ['x1'], 'y': [1]}),
        1: pd.DataFrame({'x': ['x2'], 'y': [2]}),
        2: pd.DataFrame({'x': ['x3'], 'y': [3]}),
        3: pd.DataFrame({'x': ['x4'], 'y': [4]}),
        4: pd.DataFrame({'x': ['x5'], 'y': [5]})
    }
    assert len(valid) == 0
    assert folds[0].equals(test_folds[0])


def test_euclidean_distance():
    df1 = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
    df2 = pd.DataFrame({'x': [6, 6, 3, 1, 3]})
    df3 = pd.DataFrame({'x': [1, 1, 3, 4, 5],
                        'y': [4, 4, 1, 9, 3]})
    df4 = pd.DataFrame({'x': [1, 7, 3, 2, 2],
                        'y': [6, 1, 3, 1, 3]})

    distance_one = euclidean_distance(df1, df2)
    distance_two = euclidean_distance(df3, df4)

    assert distance_one.equals(pd.Series([5, 4, 0, 3, 2]))
    assert distance_two.equals(pd.Series([2, 9, 2, 10, 3]))


def test_k_nearest_neighbor():
    train = pd.DataFrame({'x': [1, 1, 3, 4, 5],
                          'y': [4, 5, 1, 9, 3],
                          'class': [0, 1, 0, 0, 1]})
    test = pd.DataFrame({'x': [1, 7, 3, 2],
                         'y': [6, 1, 3, 1],
                         'class': [1, 1, 1, 0]})
    knn = k_nearest_neighbors(train, test, 'class', 3)
    assert knn == [0, 0, 0, 0]

