import pytest
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from mlflowgo.mlflowgo import MLFlowGo
import numpy as np
import pandas as pd


def test_ada_boost_classifier_pipeline():
    """ Test MLFlowGo with a pipeline containing AdaBoostClassifier """
    iris = load_iris()
    noise = np.random.normal(0, 0.5, iris['data'].shape)
    data = pd.DataFrame(
        data=np.c_[iris['data'] + noise,
                   iris['target']],
        columns=np.append(iris['feature_names'], ['target'])
    )

    pipeline = Pipeline([
        ('ada_boost', AdaBoostClassifier())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=data.drop(columns=['target']),
                             y=data['target'],
                             cv=-1)


def test_decision_tree_classifier_pipeline():
    """ Test MLFlowGo with a pipeline containing DecisionTreeClassifier """
    iris = load_iris()
    noise = np.random.normal(0, 0.5, iris['data'].shape)
    data = pd.DataFrame(
        data=np.c_[iris['data'] + noise,
                   iris['target']],
        columns=np.append(iris['feature_names'], ['target'])
    )

    pipeline = Pipeline([
        ('decision_tree', DecisionTreeClassifier())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=data.drop(columns=['target']),
                             y=data['target'],
                             cv=-1)


def test_gaussian_process_classifier_pipeline():
    """ Test MLFlowGo with a pipeline containing GaussianProcessClassifier """
    iris = load_iris()
    noise = np.random.normal(0, 0.5, iris['data'].shape)
    data = pd.DataFrame(
        data=np.c_[iris['data'] + noise,
                   iris['target']],
        columns=np.append(iris['feature_names'], ['target'])
    )

    pipeline = Pipeline([
        ('GPC', GaussianProcessClassifier(1.0 * RBF(1.0)))
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=data.drop(columns=['target']),
                             y=data['target'],
                             cv=-1)


def test_gradient_boosting_classifier_pipeline():
    """ Test MLFlowGo with a pipeline containing GradientBoostingClassifier """
    bc = load_breast_cancer()
    noise = np.random.normal(0, 0.5, bc['data'].shape)
    data = pd.DataFrame(
        data=np.c_[bc['data'] + noise,
                   bc['target']],
        columns=np.append(bc['feature_names'], ['target'])
    )

    pipeline = Pipeline([
        ('GBC', GradientBoostingClassifier())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=data.drop(columns=['target']),
                             y=data['target'],
                             cv=-1)


def test_knn_classifier_pipeline():
    """ Test MLFlowGo with a pipeline containing KNeighborsClassifier """
    iris = load_iris()
    noise = np.random.normal(0, 0.5, iris['data'].shape)
    data = pd.DataFrame(
        data=np.c_[iris['data'] + noise,
                   iris['target']],
        columns=np.append(iris['feature_names'], ['target'])
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('KNNC', KNeighborsClassifier())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=data.drop(columns=['target']),
                             y=data['target'],
                             cv=-1)


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
        ('logistic_regression', LogisticRegression())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=data.drop(columns=['target']),
                             y=data['target'],
                             cv=-1)


def test_gaussian_nb_pipeline():
    """ Test MLFlowGo with a pipeline containing GaussianNB """
    iris = load_iris()
    noise = np.random.normal(0, 0.5, iris['data'].shape)
    data = pd.DataFrame(
        data=np.c_[iris['data'] + noise,
                   iris['target']],
        columns=np.append(iris['feature_names'], ['target'])
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gaussian_nb', GaussianNB())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=data.drop(columns=['target']),
                             y=data['target'],
                             cv=-1)


def test_mlp_classifier_pipeline():
    """ Test MLFlowGo with a pipeline containing MLPClassifier """
    iris = load_iris()
    noise = np.random.normal(0, 0.5, iris['data'].shape)
    data = pd.DataFrame(
        data=np.c_[iris['data'] + noise,
                   iris['target']],
        columns=np.append(iris['feature_names'], ['target'])
    )

    pipeline = Pipeline([
        ('MLP', MLPClassifier())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=data.drop(columns=['target']),
                             y=data['target'],
                             cv=-1)


def test_quadratic_discriminant_analysis_pipeline():
    """ Test MLFlowGo with a pipeline containing QuadraticDiscriminantAnalysis """
    iris = load_iris()
    noise = np.random.normal(0, 0.5, iris['data'].shape)
    data = pd.DataFrame(
        data=np.c_[iris['data'] + noise,
                   iris['target']],
        columns=np.append(iris['feature_names'], ['target'])
    )

    pipeline = Pipeline([
        ('QDA', QuadraticDiscriminantAnalysis())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=data.drop(columns=['target']),
                             y=data['target'],
                             cv=-1)


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
        ('RF', RandomForestClassifier())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=data.drop(columns=['target']),
                             y=data['target'],
                             cv=-1)


def test_ridge_classifier_pipeline():
    """ Test MLFlowGo with a pipeline containing a Ridge Classifier"""
    iris = load_iris()
    noise = np.random.normal(0, 0.5, iris['data'].shape)
    data = pd.DataFrame(
        data=np.c_[iris['data'] + noise,
                   iris['target']],
        columns=np.append(iris['feature_names'], ['target'])
    )

    pipeline = Pipeline([
         ('scaler', StandardScaler()),
         ('ridge_classifier', RidgeClassifier(alpha=1.0))
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=data.drop(columns=['target']),
                             y=data['target'],
                             cv=-1)


def test_sgd_classifier_pipeline():
    """ Test MLFlowGo with a pipeline containing a SGD Classifier"""
    iris = load_iris()
    noise = np.random.normal(0, 0.5, iris['data'].shape)
    data = pd.DataFrame(
        data=np.c_[iris['data'] + noise,
                   iris['target']],
        columns=np.append(iris['feature_names'], ['target'])
    )

    pipeline = Pipeline([
         ('scaler', StandardScaler()),
         ('sgd_classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3))
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=data.drop(columns=['target']),
                             y=data['target'],
                             cv=-1)


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
        ('SVC', SVC(probability=True))
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=data.drop(columns=['target']),
                             y=data['target'],
                             cv=-1)
