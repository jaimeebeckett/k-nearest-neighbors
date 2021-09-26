import pandas as pd
import pytest
from processing import load_data, imputation, ordinal, nominal, discretization, standardization
import numpy as np


def test_load_data():
    car_file = '../data/raw/car.data'
    cancer_file = '../data/raw/breast-cancer-wisconsin.data'

    # No header specified
    car = load_data(car_file)
    cancer = load_data(cancer_file)
    assert list(car.columns) == ['vhigh', 'vhigh.1', '2', '2.1', 'small', 'low', 'unacc']
    assert list(cancer.columns) == ['1000025', '5', '1', '1.1', '1.2', '2', '1.3', '3', '1.4', '1.5', '2.1']

    car_header = load_data(car_file, header=None)
    cancer_header = load_data(cancer_file, header=None)
    assert list(car_header.columns) == [0, 1, 2, 3, 4, 5, 6]
    assert list(cancer_header.columns) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # name columns
    cancer_cols = load_data(cancer_file, names=['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
                                                'nine', 'ten'])
    assert list(cancer_cols.columns) == ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                                         'ten']

    cancer_drop = load_data(cancer_file, names=['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
                                                'nine', 'ten'], usecols=['one', 'two', 'three', 'four', 'five', 'six',
                                                                         'seven', 'eight', 'nine', 'ten'])
    assert list(cancer_drop.columns) == ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                                         'ten']

    # Missing values
    cancer = load_data(cancer_file, header=None, na_values='?')
    assert np.isnan(cancer.iloc[23, 6])


def test_imputation():
    df = pd.DataFrame({'Name': ['Smith', 'Larkin', 'Miller', 'Hayes'],
                       'Age': [22, 23, 24, pd.NA]})
    imputed_df = imputation(df)
    assert imputed_df.iloc[3, 1] == 23.0


def test_ordinal():
    df = pd.DataFrame({'Name': ['Smith', 'Larkin', 'Miller', 'Hayes'],
                       'Grade': ['Freshman', 'Sophomore', 'Junior', 'Senior']})
    ordinal_grade = ordinal(df, {'Grade': {'Freshman': 0, 'Sophomore': 1, 'Junior': 2, 'Senior': 3}})
    assert ordinal_grade['Grade'].to_list() == [0, 1, 2, 3]

def test_nominal():
    df = pd.DataFrame({'Name': ['Smith', 'Larkin', 'Miller', 'Hayes'],
                       'Favorite_Fruit': ['Apple', 'Banana', 'Orange', 'Blueberry']})
    nominal_fruit = nominal(df, ['Favorite_Fruit'])
    assert nominal_fruit['Favorite_Fruit'].to_list() == ['0001', '0010', '0100', '1000']

def test_discretization():
    df = pd.DataFrame({'Name': ['Smith', 'Larkin', 'Miller', 'Hayes'],
                       'Age': [22, 23, 24, 23]})
    bin_width = discretization(df, 'Age', 2)
    bin_frequency = discretization(df, 'Age', 2, False)

    assert bin_width['Age'].to_list() == [0, 0, 1, 0]
    assert bin_frequency['Age'].to_list() == [0, 0, 1, 1]


def test_standardization():
    df_train = pd.DataFrame({'x': ['0', '1', '2', '3'],
                             'y': [3, 4, 7, 5]})
    df_test = pd.DataFrame({'x': ['0', '1', '2', '3'],
                            'y': [4, 5, 4, 6]})
    train, test = standardization(df_train, df_test, 'y')

    assert train['y'].round(2).to_list() == [-1.02, -0.44, 1.32, 0.15]
    assert test['y'].round(2).to_list() == [-0.44, 0.15, -0.44, 0.73]
