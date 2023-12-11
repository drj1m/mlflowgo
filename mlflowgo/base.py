from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars,
    OrthogonalMatchingPursuit, BayesianRidge, ARDRegression,
    SGDRegressor, PassiveAggressiveRegressor, HuberRegressor,
    TheilSenRegressor, RANSACRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, BaggingRegressor, StackingRegressor, VotingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA
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
        Identify the model step from the pipeline
        Parameters:
            pipeline (sklearn.pipeline.Pipeline): The pipeline object
        returns:
            step_name (str): Name of the step
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
        if feature_names is None:
            feature_names = columns

        if len(columns) != len(feature_names):
            raise ValueError("length of feature names does not match number of columns")

        return feature_names

    @staticmethod
    def _get_default_metrics(task_type):
        """Returns a list of default metrics based on the task type."""
        metrics = []
        for scorer_name, scorer in _SCORERS.items():
            if task_type == CLASSIFIER_KEY and scorer._sign == 1:  # Classification metrics
                metrics.append(scorer_name)
            elif task_type == REGRESSOR_KEY and scorer._sign == -1:  # Regression metrics
                metrics.append(scorer_name)
        return metrics

    def get_model_metrics(self, metrics, task_type):
        if task_type not in [CLASSIFIER_KEY, REGRESSOR_KEY]:
            raise ValueError("Invalid model type, expected: 'classification' or 'regression'")

        if metrics is None:
            return self._get_default_metrics(task_type)

    @staticmethod
    def get_param_dist(model_name):
        """ Returns the param_dist based upon the model name
        Parameters:
            model_name(str): Name of the model from `model.__class__.__name__`
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
            }
        }

        if model_name in _param_dist:
            return _param_dist[model_name]
        else:
            return None

    @staticmethod
    def get_basic_pipeline(model_name):
        """ Returns a basic pipeline setup based upon the model name
        Parameters:
            model_name(str): Name of the model from `model.__class__.__name__`
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
            ])
        }

        if model_name in _pipeline:
            return _pipeline[model_name]
        else:
            return None
