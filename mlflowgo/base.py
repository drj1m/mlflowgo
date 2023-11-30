from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier
from sklearn.metrics._scorer import _SCORERS
from . import CLASSIFIER_KEY, REGRESSOR_KEY


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
