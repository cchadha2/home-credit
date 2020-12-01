#!/usr/bin/env python
"""Data processing script for training data used in Home-Credit model."""
import argparse

import numpy as np

import pandas as pd

IMPORTANT_FEATURES = [
    "TARGET",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

FILL_MISSING = "EXT_SOURCE_1"

CREDIT_AVERAGE = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

CREDIT_MEAN = "EXT_SOURCE_MEAN"


def fill_missing(training_data, feature):
    """Fill missing data in specified feature as mean of feature.

    Args:
        training_data (pandas.DataFrame): Training data.
        feature (str): Feature with missing values to impute.

    Returns:
        pandas.Series: Processed training data.
    """
    return training_data[feature].fillna(value=training_data[feature].mean())


def feature_averaging(training_data, features):
    """Carries out feature engineering on training data.

    Args:
        training_data (pandas.DataFrame): Training data.
        features (str): Features to average.

    Returns:
        pandas.Series: Processed training data.
    """
    return training_data[features].mean(axis=1)


def main():
    """Main function to run all steps in processing data for home-credit model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--data", type=str)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()

    with open(args.data, "rt") as training_data:
        train = pd.read_csv(filepath_or_buffer=training_data,
                            usecols=IMPORTANT_FEATURES)

    train[FILL_MISSING] = fill_missing(training_data=train, feature=FILL_MISSING)

    train[CREDIT_MEAN] = feature_averaging(training_data=train, features=CREDIT_AVERAGE)

    train.to_csv(path_or_buf=args.output, index=False)


if __name__ == "__main__":
    main()
