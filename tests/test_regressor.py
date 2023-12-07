import pytest
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlflowgo.mlflowgo import MLFlowGo
import numpy as np
import pandas as pd


def test_lasso_pipeline():
    """ Test MLFlowGo with a pipeline containing AdaBoostClassifier """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df = pd.read_csv(url, delimiter=';')

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso())
    ])
    mlflow_go = MLFlowGo(experiment_name="lasso_regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('quality', axis=1),
                             y=df['quality'], cv=-1)


def test_linear_regression_pipeline():
    """ Test MLFlowGo with a pipeline containing AdaBoostClassifier """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df = pd.read_csv(url, delimiter=';')

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    mlflow_go = MLFlowGo(experiment_name="linear_regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('quality', axis=1),
                             y=df['quality'], cv=-1)


def test_ridge_pipeline():
    """ Test MLFlowGo with a pipeline containing AdaBoostClassifier """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df = pd.read_csv(url, delimiter=';')

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge())
    ])
    mlflow_go = MLFlowGo(experiment_name="ridge_regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('quality', axis=1),
                             y=df['quality'], cv=-1)
   

