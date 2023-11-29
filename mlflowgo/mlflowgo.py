from .artifact_logger import ArtifactLogger
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier
from sklearn.metrics._scorer import _SCORERS
import subprocess
import threading
import webbrowser
import time
import pandas as pd


class MLFlowGo:
    """
    A class to integrate MLFlow with scikit-learn pipelines for data scientists.

    Attributes:
        experiment_name (str): Name of the MLFlow experiment.
        tracking_uri (str, optional): URI for MLFlow tracking server.

    Methods:
        run_experiment(model, X, y, cv, metrics, **kwargs): Runs an experiment with cross-validation and logs the results.
    """

    def __init__(self, experiment_name, tracking_uri=None):
        """
        Initialize the MLFlowGo class with experiment name and optional tracking URI.

        Parameters:
            experiment_name (str): Name of the MLFlow experiment.
            tracking_uri (str, optional): URI for MLFlow tracking server. Defaults to None.
        """
        self.experiment_name = experiment_name
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def run_experiment(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.DataFrame, cv: int = 5, **kwargs):
        """
        Runs a cross-validation experiment with the given pipeline and data, logs the metrics and model in MLFlow.

        Parameters:
            pipeline (sklearn.pipeline.Pipeline): The scikit-learn compatible pipeline to evaluate.
            X (array-like): Feature dataset.
            y (array-like): Target values.
            cv (int, optional): Number of cross-validation splits. Defaults to 5.
            metrics (list of str, optional): Metrics to log. Defaults to ['accuracy'].
            **kwargs: Additional keyword arguments for cross_val_score function.
        """
        task_type = kwargs.get('task_type', None)
        metrics = kwargs.get('metrics', None)
        feature_names = kwargs.get('feature_names', X.columns)

        if task_type is None:
            task_type = 'classification' if is_classifier(pipeline) else 'regression'

        if metrics is None:
            metrics = self._get_default_metrics(task_type)

        with mlflow.start_run(run_name=self._get_run_name(pipeline)):
            # Perform cross-validation
            cv_results = cross_val_score(pipeline, X, y, cv=cv, scoring=metrics[0], **kwargs)

            # Log parameters, metrics, and model
            self._log_params(pipeline)
            self._log_metrics(cv_results, metrics)
            if task_type == 'classification':
                self._log_artifacts(pipeline, X, y, feature_names)
            mlflow.sklearn.log_model(pipeline,
                                     self._get_model_step_from_pipeline(pipeline))

    def _log_artifacts(self, pipeline, X, y, feature_names):
        """
        Log artifacts
        """
        artifact_logger = ArtifactLogger()
        pipeline.fit(X, y)
        y_pred, y_scores = pipeline.predict(X), pipeline.predict_proba(X)

        # Plot and log ROC curve
        artifact_logger.plot_roc_curve(y,
                                       y_scores,
                                       feature_names)

        # Plot and log confusion matrix
        artifact_logger.plot_confusion_matrix(y,
                                              y_pred)

        # Save and log data sample
        artifact_logger.save_data_sample(X,
                                         100)  # Log 100 samples

        # Plot and log feature importance
        model_step = self._get_model_step_from_pipeline(pipeline)
        if hasattr(pipeline.named_steps[model_step], 'feature_importances_'):
            artifact_logger.plot_feature_importance(pipeline,
                                                    model_step,
                                                    feature_names)

    def _log_params(self, pipeline):
        """Logs the parameters of the model or pipeline."""
        if isinstance(pipeline, Pipeline):
            params = pipeline.get_params()
        else:
            params = pipeline.__dict__
        mlflow.log_params(params)

    def _log_metrics(self, cv_results, metrics):
        """Logs the metrics from cross-validation results."""
        for metric in metrics:
            mlflow.log_metric(f"mean_{metric}", cv_results.mean())
            mlflow.log_metric(f"std_{metric}", cv_results.std())

    def _get_run_name(self, pipeline):
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

    def _get_model_step_from_pipeline(self, pipeline):
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

    def _get_default_metrics(self, task_type):
        """Returns a list of default metrics based on the task type."""
        metrics = []
        for scorer_name, scorer in _SCORERS.items():
            if task_type == 'classification' and scorer._sign == 1:  # Classification metrics
                metrics.append(scorer_name)
            elif task_type == 'regression' and scorer._sign == -1:  # Regression metrics
                metrics.append(scorer_name)
        return metrics

    def run_mlflow_ui(self, port=5000, open_browser=True):
        """
        Starts the MLflow UI server on a specified port and optionally opens it in a web browser.

        Parameters:
            port (int, optional): The port to run the MLflow UI on. Defaults to 5000.
            open_browser (bool, optional): If True, opens the MLflow UI in the default web browser. Defaults to True.
        """
        def start_server():
            subprocess.run(["mlflow", "ui", "--port", str(port)])

        # Start the MLflow UI in a separate thread
        thread = threading.Thread(target=start_server)
        thread.daemon = True
        thread.start()

        if open_browser:
            # Give the server a moment to start before opening the browser
            time.sleep(2)  
            webbrowser.open(f"http://localhost:{port}")
