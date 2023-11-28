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

    def run_experiment(self, model, X, y, cv=5, **kwargs):
        """
        Runs a cross-validation experiment with the given model and data, logs the metrics and model in MLFlow.

        Parameters:
            model (estimator): The scikit-learn compatible model or pipeline to evaluate.
            X (array-like): Feature dataset.
            y (array-like): Target values.
            cv (int, optional): Number of cross-validation splits. Defaults to 5.
            metrics (list of str, optional): Metrics to log. Defaults to ['accuracy'].
            **kwargs: Additional keyword arguments for cross_val_score function.
        """
        task_type = kwargs.get('task_type', None)
        metrics = kwargs.get('metrics', None)

        if task_type is None:
            task_type = 'classification' if is_classifier(model) else 'regression'

        if metrics is None:
            metrics = self._get_default_metrics(task_type)

        with mlflow.start_run(run_name=self._get_run_name(model)):
            # Perform cross-validation
            cv_results = cross_val_score(model, X, y, cv=cv, scoring=metrics[0], **kwargs)

            # Log parameters, metrics, and model
            self._log_params(model)
            self._log_metrics(cv_results, metrics)
            mlflow.sklearn.log_model(model, "model")

    def _log_params(self, model):
        """Logs the parameters of the model or pipeline."""
        if isinstance(model, Pipeline):
            params = model.get_params()
        else:
            params = model.__dict__
        mlflow.log_params(params)

    def _log_metrics(self, cv_results, metrics):
        """Logs the metrics from cross-validation results."""
        for metric in metrics:
            mlflow.log_metric(f"mean_{metric}", cv_results.mean())
            mlflow.log_metric(f"std_{metric}", cv_results.std())

    def _get_run_name(self, model):
        """
        Generates a run name based on the model or pipeline.

        Parameters:
            model (estimator): The model or pipeline for which to generate the run name.

        Returns:
            str: Generated run name.
        """
        if isinstance(model, Pipeline):
            name = "|".join([step[0] for step in model.steps])
        else:
            name = type(model).__name__
        return name

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