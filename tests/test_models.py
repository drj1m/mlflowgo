import pytest
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from mlflowgo.mlflowgo import MLFlowGo
import numpy as np
import pandas as pd


def test_logistic_regression_pipeline():
    """ Test MLFlowGo with a pipeline containing LogisticRegression """
    iris = load_iris()
    noise = np.random.normal(0, 0.5, iris['data'].shape)
    data = pd.DataFrame(
        data=np.c_[iris['data'] + noise,
                   iris['target']],
        columns=np.append(iris['feature_names'], ['target'])
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])
    mlflow_go = MLFlowGo(experiment_name="log_reg_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=data.drop(columns=['target']),
                             y=data['target'])


def test_random_forest_pipeline():
    """ Test MLFlowGo with a pipeline containing RandomForestClassifier """
    iris = load_iris()
    noise = np.random.normal(0, 0.5, iris['data'].shape)
    data = pd.DataFrame(
        data=np.c_[iris['data'] + noise,
                   iris['target']],
        columns=np.append(iris['feature_names'], ['target'])
    )

    pipeline = Pipeline([
        ('model', RandomForestClassifier())
    ])
    mlflow_go = MLFlowGo(experiment_name="rf_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=data.drop(columns=['target']),
                             y=data['target'])


def test_svc_pipeline():
    """ Test MLFlowGo with a pipeline containing SVC """
    iris = load_iris()
    noise = np.random.normal(0, 0.5, iris['data'].shape)
    data = pd.DataFrame(
        data=np.c_[iris['data'] + noise,
                   iris['target']],
        columns=np.append(iris['feature_names'], ['target'])
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(probability=True))
    ])
    mlflow_go = MLFlowGo(experiment_name="svc_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=data.drop(columns=['target']),
                             y=data['target'])
