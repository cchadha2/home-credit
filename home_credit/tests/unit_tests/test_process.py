import unittest

import numpy as np

import pandas as pd

from home_credit.processing import feature_averaging, fill_missing


class TestProcess(unittest.TestCase):
    """Unit tests for training data processing."""

    @classmethod
    def setUpClass(cls):
        """Set up data for tests."""

        cls.df_test = pd.DataFrame(
            {
                "N_DAYS": np.random.randint(1000000, 2000000, size=50),
                "RATIO": np.random.rand(50),
                "INCOME": np.random.randint(10000, 200000, size=50),
            }
        )

        cls.df_test.iat[0, cls.df_test.columns.get_loc("N_DAYS")] = None

    def test_fill_missing(self):
        """Test if fill_missing correctly fills missing values of column with average."""
        # Verify that "N_DAYS" column contains a missing value.
        self.assertTrue(self.df_test["N_DAYS"].isna().values.any())

        # Check that fill_missing works correctly.
        result = fill_missing(training_data=self.df_test, feature="N_DAYS")
        avg_value = self.df_test["N_DAYS"].mean()
        self.assertFalse(result.isna().values.any())
        self.assertEqual(result[0], avg_value)

    def test_feature_averaging(self):
        """Test if feature_averaging correctly averages RATIO and INCOME features."""
        # Find average of "RATIO" and "INCOME" columns.
        features = ["RATIO", "INCOME"]
        avg_value = self.df_test[features].mean(axis=1)

        result = feature_averaging(training_data=self.df_test, features=features)

        pd.testing.assert_series_equal(result, avg_value)

    def test_fill_missing_with_integer_feature(self):
        """Test if fill_missing fails correctly with integer column name as input."""
        with self.assertRaises(KeyError):
            fill_missing(training_data=self.df_test, feature=5)


if __name__ == "__main__":
    unittest.main()
