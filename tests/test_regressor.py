import pytest
from sklearn.datasets import make_friedman1
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars,
    OrthogonalMatchingPursuit, BayesianRidge, ARDRegression,
    SGDRegressor, PassiveAggressiveRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor)
from mlflowgo.mlflowgo import MLFlowGo
import numpy as np
import pandas as pd


@pytest.fixture
def df():
    X, y = make_friedman1(n_samples=100, n_features=5, noise=0.1)
    feature_names = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
    df = pd.DataFrame(X, columns=feature_names)
    df['Target'] = y
    return df


def test_ada_boost_regression_pipeline(df):
    """ Test MLFlowGo with a pipeline containing Ada Boost Regression"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ada_boost_regression', AdaBoostRegressor())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('Target', axis=1),
                             y=df['Target'], cv=-1)


def test_ard_regression_pipeline(df):
    """ Test MLFlowGo with a pipeline containing ARD regression"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ard_regression', ARDRegression())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('Target', axis=1),
                             y=df['Target'], cv=-1)


def test_bayesian_ridge_pipeline(df):
    """ Test MLFlowGo with a pipeline containing Bayesian Ridge regression"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('bayesian_ridge', BayesianRidge())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('Target', axis=1),
                             y=df['Target'], cv=-1)


def test_decision_tree_regressor_pipeline(df):
    """ Test MLFlowGo with a pipeline containing Decision Tree regression"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('decision_tree', DecisionTreeRegressor())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('Target', axis=1),
                             y=df['Target'], cv=-1)


def test_elastic_net_pipeline(df):
    """ Test MLFlowGo with a pipeline containing ElasticNet regression"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('elastic_net', ElasticNet())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('Target', axis=1),
                             y=df['Target'], cv=-1)


@pytest.mark.slow
def test_extra_tree_regressor_pipeline(df):
    """ Test MLFlowGo with a pipeline containing Extra Tree Regressor"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('extra_tree_regressor', ExtraTreesRegressor())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('Target', axis=1),
                             y=df['Target'], cv=-1)


@pytest.mark.slow
def test_gaussian_process_regressor_pipeline(df):
    """ Test MLFlowGo with a pipeline containing a gaussian process regressor"""
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('GPR', GaussianProcessRegressor(kernel=kernel))
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('Target', axis=1),
                             y=df['Target'], cv=-1)


def test_gradient_boosting_regressor_pipeline(df):
    """ Test MLFlowGo with a pipeline containing a gradient boosting regressor"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gradient_boosting', GradientBoostingRegressor())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('Target', axis=1),
                             y=df['Target'], cv=-1)


@pytest.mark.slow
def test_knn_regressor_pipeline(df):
    """ Test MLFlowGo with a pipeline containing KNN regressor"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn_regressor', KNeighborsRegressor())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('Target', axis=1),
                             y=df['Target'], cv=-1)


def test_lars_pipeline(df):
    """ Test MLFlowGo with a pipeline containing LARS regression"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('LARS', Lars())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('Target', axis=1),
                             y=df['Target'], cv=-1)


def test_lasso_pipeline(df):
    """ Test MLFlowGo with a pipeline containing lasso regression"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('Target', axis=1),
                             y=df['Target'], cv=-1)


def test_lasso_lars_pipeline(df):
    """ Test MLFlowGo with a pipeline containing lasso lars regression"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso_lars', LassoLars())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('Target', axis=1),
                             y=df['Target'], cv=-1)


def test_linear_regression_pipeline(df):
    """ Test MLFlowGo with a pipeline containing linear regression """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('linear_regression', LinearRegression())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('Target', axis=1),
                             y=df['Target'], cv=-1)


def test_orthogonal_matching_pursuit_regression_pipeline(df):
    """ Test MLFlowGo with a pipeline containing linear regression """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('orthogonal_matching_pursuit', OrthogonalMatchingPursuit())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('Target', axis=1),
                             y=df['Target'], cv=-1)


def test_passive_aggressive_regressor_pipeline(df):
    """ Test MLFlowGo with a pipeline containing passive aggressive regression """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('passive_aggressive_regressor', PassiveAggressiveRegressor())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('Target', axis=1),
                             y=df['Target'], cv=-1)


def test_random_forest_regression_pipeline(df):
    """ Test MLFlowGo with a pipeline containing random forest regression """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('random_forest', RandomForestRegressor())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('Target', axis=1),
                             y=df['Target'], cv=-1)


def test_ridge_pipeline(df):
    """ Test MLFlowGo with a pipeline containing ridge regression """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('Target', axis=1),
                             y=df['Target'], cv=-1)


def test_sgr_regressor_pipeline(df):
    """ Test MLFlowGo with a pipeline containing SGR regression """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('sgr_regressor', SGDRegressor())
    ])
    mlflow_go = MLFlowGo(experiment_name="regression_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=df.drop('Target', axis=1),
                             y=df['Target'], cv=-1)
