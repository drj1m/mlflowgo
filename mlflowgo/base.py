from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars,
    OrthogonalMatchingPursuit, BayesianRidge, ARDRegression,
    SGDRegressor, PassiveAggressiveRegressor, HuberRegressor,
    TheilSenRegressor, LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (
    ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.base import is_classifier
from sklearn.metrics._scorer import _SCORERS
from . import CLASSIFIER_KEY, REGRESSOR_KEY
from scipy.stats import randint as sp_randint
from scipy.stats import uniform, loguniform


class Base():

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_pipeline(pipeline, task_type):
        """
        Get the scikit-learn pipeline and identify the modeling task type.

        Parameters:
            pipeline (sklearn.pipeline.Pipeline): The scikit-learn pipeline object.
            task_type (str or None): Type of the modeling task. If None, it will be determined based on the pipeline content.

        Returns:
            tuple: A tuple containing the scikit-learn pipeline object and the identified task type ('classification' or 'regression').

        Raises:
            ValueError: If the provided object is not a scikit-learn Pipeline.
            ValueError: If an invalid task_type is provided.
        """
        if not isinstance(pipeline, Pipeline):
            raise ValueError("The provided object is not a scikit-learn Pipeline.")

        if task_type is None:
            task_type = CLASSIFIER_KEY if is_classifier(pipeline) else REGRESSOR_KEY
        elif task_type not in [CLASSIFIER_KEY, REGRESSOR_KEY]:
            raise ValueError("Invalid model type, expected: 'classification' or 'regression'")

        return pipeline, task_type

    @staticmethod
    def get_model_step_from_pipeline(pipeline):
        """
        Identify the model step from the pipeline.

        Parameters:
            pipeline (sklearn.pipeline.Pipeline): The pipeline object.

        Returns:
            str: Name of the step that represents the model.
        """
        if not isinstance(pipeline, Pipeline):
            raise ValueError("The provided object is not a scikit-learn Pipeline.")

        for step_name, step in pipeline.steps:
            # A model is expected to have a 'predict' method
            if hasattr(step, 'predict'):
                return step_name

    @staticmethod
    def get_run_name(pipeline):
        """
        Generates a run name based on the model or pipeline.

        Parameters:
            pipeline (sklearn.pipeline.Pipeline): The pipeline for which to generate the run name.

        Returns:
            str: Generated run name.
        """
        if isinstance(pipeline, Pipeline):
            name = "|".join([step[0] for step in pipeline.steps])
        else:
            name = type(pipeline).__name__
        return name

    @staticmethod
    def get_feature_names(feature_names, columns):
        """
        Get feature names and validate their length.

        Parameters:
            feature_names (list or None): List of feature names.
            columns (list): List of column names in the dataset.

        Returns:
            list: List of feature names to use.

        Raises:
            ValueError: If the length of feature names does not match the number of columns.
        """
        if feature_names is None:
            feature_names = columns

        if len(columns) != len(feature_names):
            raise ValueError("length of feature names does not match number of columns")

        return feature_names

    @staticmethod
    def _get_default_metrics(task_type):
        """
        Get a list of default evaluation metrics based on the task type.

        Parameters:
            task_type (str): Type of the modeling task, 'classification' or 'regression'.

        Returns:
            list: List of default evaluation metrics corresponding to the task type.

        Raises:
            ValueError: If an invalid task_type is provided.
        """
        metrics = []
        for scorer_name, scorer in _SCORERS.items():
            if task_type == CLASSIFIER_KEY and scorer._sign == 1:  # Classification metrics
                metrics.append(scorer_name)
            elif task_type == REGRESSOR_KEY and scorer._sign == -1:  # Regression metrics
                metrics.append(scorer_name)
        return metrics

    def get_model_metrics(self, metrics, task_type):
        """
        Get the list of model evaluation metrics based on task type.

        Parameters:
            metrics (list or None): List of specified metrics or None to use default metrics.
            task_type (str): Type of the modeling task, 'classification' or 'regression'.

        Returns:
            list: List of model evaluation metrics.

        Raises:
            ValueError: If an invalid task_type is provided.
        """
        if task_type not in [CLASSIFIER_KEY, REGRESSOR_KEY]:
            raise ValueError("Invalid model type, expected: 'classification' or 'regression'")

        if metrics is None:
            return self._get_default_metrics(task_type)
        else:
            return metrics

    @staticmethod
    def get_param_dist(model_name):
        """
        Get parameter distribution based on the model name.

        Parameters:
            model_name (str): Name of the model from `model.__class__.__name__`.

        Returns:
            dict or None: Parameter distribution dictionary or None if not found for the model.
        """
        _param_dist = {
            'AdaBoostRegressor': {
                "n_estimators": sp_randint(50, 500),
                "learning_rate": uniform(0.01, 1.0),
                "loss": ['linear', 'square', 'exponential']
            },
            'ARDRegression': {
                "n_iter": sp_randint(100, 500),
                "alpha_1": uniform(1e-6, 1e-5),
                "alpha_2": uniform(1e-6, 1e-5),
                "lambda_1": uniform(1e-6, 1e-5),
                "lambda_2": uniform(1e-6, 1e-5)
            },
            'BayesianRidge': {
                "n_iter": sp_randint(100, 500),
                "alpha_1": uniform(1e-6, 1e-5),
                "alpha_2": uniform(1e-6, 1e-5),
                "lambda_1": uniform(1e-6, 1e-5),
                "lambda_2": uniform(1e-6, 1e-5)
            },
            'DecisionTreeRegressor': {
                "max_depth": [None, 10, 20, 30, 40],
                "min_samples_split": sp_randint(2, 10),
                "min_samples_leaf": sp_randint(1, 10)
            },
            'ElasticNet': {
                "alpha": uniform(0.1, 10),
                "l1_ratio": uniform(0.0, 1.0)
            },
            'ExtraTreesRegressor': {
                "n_estimators": sp_randint(100, 500),
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": sp_randint(2, 10),
                "min_samples_leaf": sp_randint(1, 10)
            },
            'GradientBoostingRegressor': {
                "n_estimators": sp_randint(100, 500),
                "learning_rate": uniform(0.01, 0.2),
                "max_depth": sp_randint(3, 10)
            },
            'KNeighborsRegressor': {
                "n_neighbors": sp_randint(1, 30),
                "weights": ['uniform', 'distance'],
                "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']
            },
            'HuberRegressor': {
                "epsilon": uniform(1.0, 3.0),
                "alpha": uniform(0.00001, 0.1)
            },
            'Lars': {
                "n_nonzero_coefs": sp_randint(1, 20)
            },
            'Lasso': {
                "alpha": uniform(0.1, 10)
            },
            'LassoLars': {
                "alpha": uniform(0.01, 10)
            },
            'LinearRegression': {
                'fit_intercept': [True, False],
                'positive': [True, False]
            },
            'MLPRegressor': {
                "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                "activation": ['tanh', 'relu'],
                "solver": ['sgd', 'adam'],
                "alpha": uniform(0.0001, 0.05),
                "learning_rate": ['constant', 'adaptive']
            },
            'OrthogonalMatchingPursuit': {
                "n_nonzero_coefs": sp_randint(1, 20)
            },
            'PassiveAggressiveRegressor': {
                "max_iter": sp_randint(1000, 5000),
                "tol": loguniform(1e-4, 1e-1),
                "C": uniform(0.1, 10)
            },
            'RandomForestRegressor': {
                "n_estimators": sp_randint(10, 200),
                "max_depth": [3, None],
                "max_features": sp_randint(1, 11),
                "min_samples_split": sp_randint(2, 11),
                "min_samples_leaf": sp_randint(1, 11),
                "bootstrap": [True, False]
            },
            'Ridge': {
                "alpha": uniform(0.1, 10)
            },
            'SGDRegressor': {
                "max_iter": sp_randint(1000, 5000),
                "tol": loguniform(1e-4, 1e-1),
                "penalty": ['l2', 'l1', 'elasticnet'],
                "alpha": uniform(0.0001, 0.1)
            },
            'SVR': {
                "C": uniform(0.1, 10),
                "kernel": ['linear', 'poly', 'rbf', 'sigmoid']
            },
            'TheilSenRegressor': {
                "max_subpopulation": sp_randint(10, 500)
            },
            'AdaBoostClassifier': {
                "n_estimators": sp_randint(50, 500),
                "learning_rate": uniform(0.01, 1.0)
            },
            'DecisionTreeClassifier': {
                "max_depth": [None, 10, 20, 30, 40],
                "min_samples_split": sp_randint(2, 10),
                "min_samples_leaf": sp_randint(1, 10)
            },
            'ExtraTreesClassifier': {
                "n_estimators": sp_randint(100, 500),
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": sp_randint(2, 10),
                "min_samples_leaf": sp_randint(1, 10),
                "bootstrap": [True, False]
            },
            'GradientBoostingClassifier': {
                "n_estimators": sp_randint(100, 500),
                "learning_rate": uniform(0.01, 0.2),
                "max_depth": sp_randint(3, 10)
            },
            'KNeighborsClassifier': {
                "n_neighbors": sp_randint(1, 30),
                "weights": ['uniform', 'distance'],
                "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']
            },
            'LinearDiscriminantAnalysis': {
                "solver": ['svd', 'lsqr', 'eigen']
            },
            'LogisticRegression': {
                "C": uniform(0.1, 10),
                "penalty": ['l2', 'l1', 'elasticnet'],
                "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            },
            'MLPClassifier': {
                "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                "activation": ['tanh', 'relu'],
                "solver": ['sgd', 'adam'],
                "alpha": uniform(0.0001, 0.05),
                "learning_rate": ['constant', 'adaptive']
            },
            'Perceptron': {
                "penalty": [None, 'l2', 'l1', 'elasticnet'],
                "alpha": uniform(0.0001, 0.01),
                "max_iter": sp_randint(1000, 5000),
                "eta0": uniform(0.1, 1)
            },
            'QuadraticDiscriminantAnalysis': {
                "reg_param": uniform(0, 1)
            },
            'RandomForestClassifier': {
                "n_estimators": sp_randint(10, 200),
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": sp_randint(2, 11),
                "min_samples_leaf": sp_randint(1, 11),
                "bootstrap": [True, False]
            },
            'RidgeClassifier': {
                "alpha": uniform(0.1, 10),
                "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            },
            'SGDClassifier': {
                "max_iter": sp_randint(1000, 5000),
                "tol": uniform(1e-4, 1e-1),
                "penalty": ['l2', 'l1', 'elasticnet'],
                "alpha": uniform(0.0001, 0.1)
            },
            'SVC': {
                "C": uniform(0.1, 10),
                "kernel": ['linear', 'poly', 'rbf', 'sigmoid']
            }
        }

        if model_name in _param_dist:
            return _param_dist[model_name]
        else:
            return None

    @staticmethod
    def get_basic_pipeline(model_name):
        """
        Get a basic pipeline setup based on the model name.

        Parameters:
            model_name (str): Name of the model from `model.__class__.__name__`.

        Returns:
            sklearn.pipeline.Pipeline or None: Basic pipeline or None if not found for the model.
        """
        _pipeline = {
            'AdaBoostRegressor': Pipeline([
                ('scaler', StandardScaler()),
                ('ada_boost_regression', AdaBoostRegressor())
            ]),
            'ARDRegression': Pipeline([
                ('scaler', StandardScaler()),
                ('ard_regression', ARDRegression())
            ]),
            'BayesianRidge': Pipeline([
                ('scaler', StandardScaler()),
                ('bayesian_ridge', BayesianRidge())
            ]),
            'DecisionTreeRegressor': Pipeline([
                ('scaler', StandardScaler()),
                ('decision_tree', DecisionTreeRegressor())
            ]),
            'ElasticNet': Pipeline([
                ('scaler', StandardScaler()),
                ('elastic_net', ElasticNet())
            ]),
            'ExtraTreesRegressor': Pipeline([
                ('scaler', StandardScaler()),
                ('extra_tree_regressor', ExtraTreesRegressor())
            ]),
            'GradientBoostingRegressor': Pipeline([
                ('scaler', StandardScaler()),
                ('GPR', GradientBoostingRegressor())
            ]),
            'KNeighborsRegressor': Pipeline([
                ('scaler', StandardScaler()),
                ('knn_regressor', KNeighborsRegressor())
            ]),
            'HuberRegressor': Pipeline([
                ('scaler', StandardScaler()),
                ('hubber_regressor', HuberRegressor(epsilon=1.35))
            ]),
            'Lars': Pipeline([
                ('scaler', StandardScaler()),
                ('LARS', Lars())
            ]),
            'Lasso': Pipeline([
                ('scaler', StandardScaler()),
                ('lasso', Lasso())
            ]),
            'LassoLars': Pipeline([
                ('scaler', StandardScaler()),
                ('lasso_lars', LassoLars())
            ]),
            'LinearRegression': Pipeline([
                ('scaler', StandardScaler()),
                ('linear_regression', LinearRegression())
            ]),
            'MLPRegressor': Pipeline([
                ('scaler', StandardScaler()),
                ('mlp_regressor', MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam'))
            ]),
            'OrthogonalMatchingPursuit': Pipeline([
                ('scaler', StandardScaler()),
                ('orthogonal_matching_pursuit', OrthogonalMatchingPursuit())
            ]),
            'PassiveAggressiveRegressor': Pipeline([
                ('scaler', StandardScaler()),
                ('passive_aggressive_regressor', PassiveAggressiveRegressor())
            ]),
            'RandomForestRegressor': Pipeline([
                ('scaler', StandardScaler()),
                ('random_forest', RandomForestRegressor())
            ]),
            'Ridge': Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge())
            ]),
            'SGDRegressor': Pipeline([
                ('scaler', StandardScaler()),
                ('sgr_regressor', SGDRegressor())
            ]),
            'SVR': Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR())
            ]),
            'TheilSenRegressor': Pipeline([
                ('scaler', StandardScaler()),
                ('theil_sen', TheilSenRegressor())
            ]),
            'AdaBoostClassifier': Pipeline([
                ('ada_boost', AdaBoostClassifier())
            ]),
            'DecisionTreeClassifier': Pipeline([
                ('decision_tree', DecisionTreeClassifier())
            ]),
            'ExtraTreesClassifier': Pipeline([
                ('scaler', StandardScaler()),
                ('extra_trees', ExtraTreesClassifier(n_estimators=100, max_depth=None))
            ]),
            'GradientBoostingClassifier': Pipeline([
                ('GBC', GradientBoostingClassifier())
            ]),
            'KNeighborsClassifier': Pipeline([
                ('scaler', StandardScaler()),
                ('KNNC', KNeighborsClassifier())
            ]),
            'LinearDiscriminantAnalysis': Pipeline([
                ('scaler', StandardScaler()),
                ('lda', LinearDiscriminantAnalysis())
            ]),
            'LogisticRegression': Pipeline([
                ('scaler', StandardScaler()),
                ('logistic_regression', LogisticRegression())
            ]),
            'MLPClassifier': Pipeline([
                ('scaler', StandardScaler()),
                ('MLP', MLPClassifier())
            ]),
            'Perceptron': Pipeline([
                ('scaler', StandardScaler()),
                ('perceptron', Perceptron(max_iter=1000, tol=1e-3, eta0=1.0, penalty='l2'))
            ]),
            'QuadraticDiscriminantAnalysis': Pipeline([
                ('QDA', QuadraticDiscriminantAnalysis())
            ]),
            'RandomForestClassifier': Pipeline([
                ('RF', RandomForestClassifier())
            ]),
            'RidgeClassifier': Pipeline([
                ('scaler', StandardScaler()),
                ('ridge_classifier', RidgeClassifier(alpha=1.0))
            ]),
            'SGDClassifier': Pipeline([
                ('scaler', StandardScaler()),
                ('sgd_classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3))
            ]),
            'SVC': Pipeline([
                ('scaler', StandardScaler()),
                ('SVC', SVC(probability=True))
            ])
        }

        if model_name in _pipeline:
            return _pipeline[model_name]
        else:
            return None
