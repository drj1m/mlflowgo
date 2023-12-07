import pytest
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, Lars
from mlflowgo.mlflowgo import MLFlowGo
import numpy as np
import pandas as pd


def test_elastic_net_pipeline():
    """ Test MLFlowGo with a pipeline containing ElasticNet regression"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df = pd.read_csv(url, delimiter=';')

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('elastic_net', ElasticNet())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('quality', axis=1),
                             y=df['quality'], cv=-1)


def test_lars_pipeline():
    """ Test MLFlowGo with a pipeline containing LARS regression"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df = pd.read_csv(url, delimiter=';')

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('LARS', Lars())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('quality', axis=1),
                             y=df['quality'], cv=-1)


def test_lasso_pipeline():
    """ Test MLFlowGo with a pipeline containing lasso regression"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df = pd.read_csv(url, delimiter=';')

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('quality', axis=1),
                             y=df['quality'], cv=-1)


def test_linear_regression_pipeline():
    """ Test MLFlowGo with a pipeline containing linear regression """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df = pd.read_csv(url, delimiter=';')

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('linear_regression', LinearRegression())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('quality', axis=1),
                             y=df['quality'], cv=-1)


def test_ridge_pipeline():
    """ Test MLFlowGo with a pipeline containing ridge regression """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df = pd.read_csv(url, delimiter=';')

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('quality', axis=1),
                             y=df['quality'], cv=-1)
   

