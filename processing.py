import pandas as pd
import numpy as np
import math
import itertools


def load_data(filename, transform={}, **kwargs):
    df = pd.read_csv(filename, **kwargs)
    for col, func in transform.items():
        df[col] = func(df[col] + 1)
    return df


def imputation(df, col=None, fill_value=0):
    if col:
        df[col] = df[col].fillna()
    df = df.fillna(df.mean())
    return df


def ordinal(df, col_order):
    for col, order in col_order.items():
        df[col] = df[col].replace(order)
    return df


def nominal(df, columns):
    unique_len = 0
    values = []
    for col in columns:
        unique = df[col].unique()
        if len(unique) != unique_len:
            unique_len = len(unique)
            values = [format(2 ** i, f"0{unique_len}b") for i in range(unique_len)]
            one_hot = dict(zip(unique, values))
        else:
            one_hot = dict(zip(unique, values))

        df[col] = df[col].replace(one_hot)
    return df


def discretization(df, col, bins, width=True):
    if width:
        unique_bins = {df[col].min(): 0}
        cutoff = (df[col].max() - df[col].min()) // bins
        for i in range(bins):
            unique_values = df[(df[col] > (i * cutoff) + df[col].min()) &
                               (df[col] <= ((i + 1) * cutoff) + df[col].min())][col].unique()
            unique_bins.update(dict(zip(unique_values, itertools.repeat(i))))
        df[col] = df[col].replace(unique_bins)
    else:
        bin_col = []
        bin_size = len(df) // bins
        left_over = len(df) % bins
        for i in range(bins):
            bin_col.extend([i] * bin_size)
            if left_over > 0:
                bin_col.append(i)
                left_over -= 1
        df = df.sort_values(col)
        df[col] = bin_col
    return df


def standardization(train_df, test_df, col):
    mu = train_df[col].mean()
    sigma = train_df[col].std()

    train_df[col] = (train_df[col] - mu) / sigma
    test_df[col] = (test_df[col] - mu) / sigma

    return train_df, test_df
