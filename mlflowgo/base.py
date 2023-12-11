from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier
from sklearn.metrics._scorer import _SCORERS
from . import CLASSIFIER_KEY, REGRESSOR_KEY
from scipy.stats import randint as sp_randint
from scipy.stats import uniform


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
            'GradientBoostingRegressor': {
                "n_estimators": sp_randint(100, 500),
                "learning_rate": uniform(0.01, 0.2),
                "max_depth": sp_randint(3, 10)
            },
            'Lasso': {
                "alpha": uniform(0.1, 10)
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
            'SVR': {
                "C": uniform(0.1, 10),
                "kernel": ['linear', 'poly', 'rbf', 'sigmoid']
            },
        }

        if model_name in _param_dist:
            return _param_dist[model_name]
        else:
            return None
