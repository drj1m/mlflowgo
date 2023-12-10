from typing import Dict

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

from mlflowgo.cross_validation import TemporalCrossValidation


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    # Generating a time series with static dates and times
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 10)

    # Creating a time series DataFrame with additional columns
    time_series_data = {
        'Date': [],
        'Time': [],
    }

    # Generate 2-4 rows with the same date and different times
    for _ in range(2, 30):  # Set the desired number of rows (2-4)
        date = start_date + timedelta(days=(_ - 2))  # Increment the date by the index
        time_series_data['Date'].extend([date.date()] * _)  # Use date() to extract only the date part
        time_series_data['Time'].extend(
            pd.date_range(datetime(date.year, date.month, date.day),
                          periods=_, freq='H').time)

    time_series_df = pd.DataFrame(time_series_data)

    time_series_df = time_series_df.sort_values(by=['Date', 'Time'])

    return time_series_df


@pytest.fixture
def temporal_class(
    sample_dataframe
) -> TemporalCrossValidation:

    return TemporalCrossValidation(
        df=sample_dataframe,
        temporal_column='Date',
    )


def test_temporal_strategies_5_folds_cv_1(
    temporal_class
):

    temporal_strategies = temporal_class.compile_strategies(n_cvs=5)

    cv_1 = temporal_strategies['CV_1']

    assert (
        cv_1['train'] == [
            date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)]
    ).all()
    assert (cv_1['val'] == [date(2023, 1, 4)]).all()
    assert (cv_1['test'] == [date(2023, 1, 5)]).all()


def test_temporal_strategies_5_folds_cv_3(
    temporal_class
):

    temporal_strategies = temporal_class.compile_strategies(n_cvs=5)

    cv_1 = temporal_strategies['CV_3']

    assert (
        cv_1['train'] == [
            date(2023, 1, 12), date(2023, 1, 13), date(2023, 1, 14)]
    ).all()
    assert (cv_1['val'] == [date(2023, 1, 15)]).all()
    assert (cv_1['test'] == [date(2023, 1, 16)]).all()


def test_temporal_strategies_2_folds_cv_1(
    temporal_class
):

    temporal_strategies = temporal_class.compile_strategies(n_cvs=2)

    cv_1 = temporal_strategies['CV_1']

    assert (cv_1['train'] == [
        date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3),
        date(2023, 1, 4), date(2023, 1, 5), date(2023, 1, 6),
        date(2023, 1, 7), date(2023, 1, 8), date(2023, 1, 9),
        date(2023, 1, 10),
    ]).all()
    assert (cv_1['val'] == [date(2023, 1, 11), date(2023, 1, 12)]).all()
    assert (cv_1['test'] == [date(2023, 1, 13), date(2023, 1, 14)]).all()
