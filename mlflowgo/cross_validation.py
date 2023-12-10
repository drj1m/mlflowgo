from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class TemporalCrossValidation:
    """
    Class for performing temporal cross-validation on time series data.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the time series data.
        temporal_column(str): The name of the column containing temporal values.
        temporal_order_columns (Optional[List[str]]) = List of temporal columns to order the data by
        train_th (Optional[float]): Percentage of data to be used for training.
        val_th (Optional[float]): Percentage of data to be used for validation.
        test_th (Optional[float]): Percentage of data to be used for testing.

    Methods:
    - _get_split_indices(unique_temporals: List[Any], start_idx: int, end_idx: int) -> Tuple[int, int, int]
        Helper method to get train, validation, and test indices for a rolling window.
    - rolling_temporal_cv(unique_temporals: List[Any], n: int) -> List[Tuple[int, int, int]]
        Perform rolling temporal cross-validation for 'n' temporal CVs.
    - compile_strategies(n_cvs: int) -> Dict[str, Dict[str, str]]
        Compile temporal cross-validation strategies.
    - build_strategy(cv: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Build a specific temporal cross-validation strategy.

    Raises:
    - ValueError
        If the sum of train/val/test thresholds is not equal to 1.
    """

    TRAIN_TH: float = 0.75
    VAL_TH: float = 0.15
    TEST_TH: float = 0.10

    def __init__(
        self,
        df: pd.DataFrame,
        temporal_column: str,
        temporal_order_columns: Optional[List[str]] = None,
        train_th: Optional[float] = TRAIN_TH,
        val_th: Optional[float] = VAL_TH,
        test_th: Optional[float] = TEST_TH,
    ):
        self.df = df
        self.temporal_column = temporal_column

        if temporal_order_columns:
            self.df.sort_values(by=temporal_order_columns, inplace=True)

        self.train_th = train_th
        self.val_th = val_th
        self.test_th = test_th

        if (self.train_th + self.val_th + self.test_th) != 1:
            raise ValueError("The sum of the train/val/test thresholds must equate to 1.")

        self.strategy_dict = {}

    def _get_split_indices(
        self, start_idx: int, end_idx: int
    ) -> Tuple[int, int, int]:
        """
        Helper method to get train, validation, and test indices for a rolling window.

        Parameters:
            unique_temporals (List[Any]): List of unique temporal data points.
            start_idx (int): Starting index of the rolling window.
            end_idx (int): Ending index of the rolling window.

        Returns:
        - Tuple of train, validation, and test indices.
        """
        indices_window = list(range(start_idx, end_idx))
        min_data_points = max(len(indices_window) * self.train_th, 1)

        if len(indices_window) < min_data_points:
            raise ValueError("Not enough data points to satisfy the percentage splits.")

        train_indices, temp_indices = train_test_split(
            indices_window, test_size=1 - self.train_th,
            shuffle=False,
            random_state=42
        )
        val_test_indices = list(set(indices_window) - set(train_indices))

        val_indices, test_indices = train_test_split(
            val_test_indices,
            test_size=self.test_th / (self.val_th + self.test_th),
            shuffle=False, random_state=42
        )

        return train_indices, val_indices, test_indices

    def rolling_temporal_cv(self, unique_temporals: List[Any], n: int) -> List[Tuple[int, int, int]]:
        """
        Perform rolling temporal cross-validation for 'n' temporal CVs.

        Parameters:
            unique_temporals (List[Any]): List of unique temporal data points.
            n (int): Number of temporal cross-validation sets to build.

        Returns:
            List of tuples, where each tuple contains train, validation, and test indices.
        """
        temporal_cvs = []
        total_items = len(unique_temporals)

        for i in range(n):
            start_idx = int((i / n) * total_items)
            end_idx = int(((i + 1) / n) * total_items)

            train_indices, val_indices, test_indices = self._get_split_indices(
                start_idx, end_idx)
            temporal_cvs.append((train_indices, val_indices, test_indices))

        return temporal_cvs

    def compile_strategies(self, n_cvs: int) -> Dict[str, Dict[str, str]]:
        """
        Compile temporal cross-validation strategies.

        Parameters:
            n_cvs (int): Number of temporal cross-validation strategies to compile.

        Returns:
            Dictionary of temporal cross-validation strategies.
        """
        unique_temporals = np.sort(self.df[self.temporal_column].unique())
        temporal_cvs = self.rolling_temporal_cv(unique_temporals, n_cvs)

        for i, (train_indices, val_indices, test_indices) in enumerate(temporal_cvs):
            self.strategy_dict[f"CV_{i + 1}"] = {
                "train": unique_temporals[train_indices],
                "val": unique_temporals[val_indices],
                "test": unique_temporals[test_indices],
            }

        return self.strategy_dict

    def build_strategy(self, cv: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Build a specific temporal cross-validation strategy.

        Parameters:
            cv (str): The name of the temporal cross-validation strategy to build.

        Returns:
            Tuple of DataFrames containing train, validation, and test data.
        """
        strategy = self.strategy_dict[cv]

        train_df = self.df[self.df[self.temporal_column].isin(strategy['train'])]
        val_df = self.df[self.df[self.temporal_column].isin(strategy['val'])]
        test_df = self.df[self.df[self.temporal_column].isin(strategy['test'])]

        return train_df, val_df, test_df
