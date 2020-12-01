#!/usr/bin/env python
"""Training script for Home-Credit model."""
import argparse

import joblib

from lightgbm import Dataset, train

import pandas as pd

MODEL_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 20,
    "colsample_bytree": 0.9497036,
    "subsample": 0.8715623,
    "subsample_freq": 1,
    "max_depth": 8,
    "reg_alpha": 0.041545473,
    "reg_lambda": 0.0735294,
    "min_split_gain": 0.0222415,
    "min_child_weight": 60,
    "seed": 0,
    "verbose": -1,
}

LABEL = "TARGET"


def lightgbm_trainer(training_data, label, model_params):
    """Train LightGBM model on training data.

    Args:
        training_data (lightgbm.Dataset): Training data.
        label (str): Target column in training data.
        model_params (dict): Training parameters.

    Returns:
        lightgbm.Booster: Trained LightGBM model.
    """
    training_data = Dataset(data=training_data.drop(label, axis=1), label=training_data[LABEL])
    return train(train_set=training_data, params=model_params)


def main():
    """Main function to run all steps in training model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--data", type=str)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()

    with open(parser.parse_args().data, "rt") as training_data:
        train_df = pd.read_csv(training_data)

    lgb_classifier = lightgbm_trainer(training_data=train_df, label=LABEL, model_params=MODEL_PARAMS)

    joblib.dump(value=lgb_classifier, filename=args.output)


if __name__ == "__main__":
    main()
