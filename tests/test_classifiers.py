import pytest
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier,
    BaggingClassifier, VotingClassifier, StackingClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from mlflowgo.mlflowgo import MLFlowGo
import numpy as np
import pandas as pd


@pytest.fixture
def iris():
    iris = load_iris()
    noise = np.random.normal(0, 0.5, iris['data'].shape)
    data = pd.DataFrame(
        data=np.c_[iris['data'] + noise,
                   iris['target']],
        columns=np.append(iris['feature_names'], ['target'])
    )
    return data


def test_ada_boost_classifier_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing AdaBoostClassifier """
    pipeline = Pipeline([
        ('ada_boost', AdaBoostClassifier())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_bagging_classifier_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing Bagging Classifier """
    base_estimator = DecisionTreeClassifier()
    pipeline = Pipeline([
         ('bagging_classifier', BaggingClassifier(base_estimator=base_estimator, n_estimators=10)) 
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_bernoulli_nb_classifier_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing BernoulliNB Classifier """
    pipeline = Pipeline([
         ('bernoulli_nb', BernoulliNB(force_alpha=True))
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_decision_tree_classifier_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing DecisionTreeClassifier """
    pipeline = Pipeline([
        ('decision_tree', DecisionTreeClassifier())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_extra_trees_classifier_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing Extra Trees Classifier """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('extra_trees', ExtraTreesClassifier(n_estimators=100, max_depth=None))
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_gaussian_process_classifier_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing GaussianProcessClassifier """
    pipeline = Pipeline([
        ('GPC', GaussianProcessClassifier(1.0 * RBF(1.0)))
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
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


def test_knn_classifier_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing KNeighborsClassifier """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('KNNC', KNeighborsClassifier())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_label_propagation_classifier_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing lable propagation """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('label_propagation', LabelPropagation())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_label_spreading_classifier_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing label spreading """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('label_spreading', LabelSpreading())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_linear_svc_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing a linear SVC"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('linear_svc', LinearSVC())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_lgbm_classifier_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing LGBM Classifier """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lgbm', LGBMClassifier())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_linear_discriminant_analysis_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing Linear Discriminant Analysis"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_logistic_regression_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing LogisticRegression """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic_regression', LogisticRegression())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_gaussian_nb_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing GaussianNB """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gaussian_nb', GaussianNB())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_mlp_classifier_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing MLPClassifier """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('MLP', MLPClassifier())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_nearest_centroid_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing Nearest Centroid Classifier """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('nearest_centroid', NearestCentroid())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_nusvc_classifier_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing NuSVC """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('NuSVC', NuSVC())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_passive_aggresive_classifier_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing Perceptron Classifier"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('passive_aggressive', PassiveAggressiveClassifier(max_iter=1000, tol=1e-3))
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_perceptron_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing Perceptron Classifier"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('perceptron', Perceptron(max_iter=1000, tol=1e-3, eta0=1.0, penalty='l2'))
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_quadratic_discriminant_analysis_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing QuadraticDiscriminantAnalysis """
    pipeline = Pipeline([
        ('QDA', QuadraticDiscriminantAnalysis())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_random_forest_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing RandomForestClassifier """
    pipeline = Pipeline([
        ('RF', RandomForestClassifier())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_ridge_classifier_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing a Ridge Classifier"""
    pipeline = Pipeline([
         ('scaler', StandardScaler()),
         ('ridge_classifier', RidgeClassifier(alpha=1.0))
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_sgd_classifier_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing a SGD Classifier"""
    pipeline = Pipeline([
         ('scaler', StandardScaler()),
         ('sgd_classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3))
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_stacking_classifier_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing a Stacking Classifier"""
    base_estimators = [
        ('svm', SVC(probability=True)),
        ('decision_tree', DecisionTreeClassifier())
    ]
    final_estimator = LogisticRegression()
    pipeline = Pipeline([
         ('scaler', StandardScaler()),
         ('stacking_classifier', StackingClassifier(estimators=base_estimators, final_estimator=final_estimator))
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_svc_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing SVC """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('SVC', SVC(probability=True))
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_voting_classifier_pipeline(iris):
    """ Test MLFlowGo with a pipeline containing a voting classifier"""
    base_estimators = [
        ('logistic_regression', LogisticRegression()),
        ('decision_tree', DecisionTreeClassifier()),
        ('svm', SVC(probability=True))
    ]
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('voting_classifier', VotingClassifier(estimators=base_estimators, voting='soft'))
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)


def test_xgboost_classifier_pipeline(iris):
    """ Test MLFlowGo with a pipeline contianing XGBoost"""

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgboost_classifier', XGBClassifier())
    ])
    mlflow_go = MLFlowGo(experiment_name="classification_test")
    mlflow_go.run_experiment(pipeline=pipeline,
                             X=iris.drop(columns=['target']),
                             y=iris['target'],
                             cv=-1)
