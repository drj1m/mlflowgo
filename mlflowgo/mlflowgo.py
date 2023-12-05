from .artifact_base import ArtifactBase
from .regressor import Regressor
from .classifier import Classifier
from . import CLASSIFIER_KEY, REGRESSOR_KEY
from .base import Base
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
import subprocess
import threading
import webbrowser
import requests
import time
import pandas as pd
import numpy as np


class MLFlowGo(Base):
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
            X (pd.DataFrame): Feature dataset.
            y (pd.DataFrame): Target values.
            cv (int, optional): Number of cross-validation splits. Defaults to 5.
            metrics (list of str, optional): Metrics to log. Defaults to ['accuracy'].
            **kwargs: Additional keyword arguments for cross_val_score function.
        """

        if not isinstance(pipeline, (list, tuple, np.ndarray)):
            pipeline = [pipeline]

        for _pipeline in pipeline:
            self.pipeline, self.task_type = self.get_pipeline(
                _pipeline,
                kwargs.get('task_type', None))
            self.metrics = self.get_model_metrics(
                kwargs.get('metrics', None),
                self.task_type)
            self.feature_names = self.get_feature_names(
                kwargs.get('feature_names', None),
                X.columns
            )
            self.param_name = kwargs.get('param_name', None)
            self.param_range = kwargs.get('param_range', None)
            self.objective = kwargs.get('objective', None)
            self.dataset_desc = kwargs.get('dataset_desc', None)

            self.model_step = self.get_model_step_from_pipeline(self.pipeline)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.33)

            with mlflow.start_run(run_name=self.get_run_name(self.pipeline)):
                # Perform cross-validation
                if cv != -1:
                    cv_results = [cross_val_score(
                        self.pipeline,
                        self.X_train,
                        self.y_train,
                        cv=cv,
                        scoring=m) for m in self.metrics]
                else:
                    cv_results = None

                # Log parameters, metrics, and model
                self._log_params()
                if cv_results is not None: self._log_metrics(cv_results, self.metrics)
                self._log_artifacts()
                mlflow.sklearn.log_model(self.pipeline,
                                         self.model_step)

    def _log_artifacts(self):
        """
        Log all relevant artifacts for the experiment
        """

        base = ArtifactBase(
            pipeline=self.pipeline.fit(self.X_train, self.y_train),
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test,
            model_step=self.model_step,
            feature_names=self.feature_names,
            metric=self.metrics[0],
            param_name=self.param_name,
            param_range=self.param_range,
            objective=self.objective,
            dataset_desc=self.dataset_desc
        )

        if self.task_type == REGRESSOR_KEY:
            artifact_logger = Regressor(base)
        elif self.task_type == CLASSIFIER_KEY:
            artifact_logger = Classifier(base)

        artifact_logger.log()

    def _log_params(self):
        """
        Logs the parameters of the model or pipeline.
        """
        if isinstance(self.pipeline, Pipeline):
            params = self.pipeline.get_params()
        else:
            params = self.pipeline.__dict__
        mlflow.log_params(params)

    def _log_metrics(self, cv_results, metrics):
        """Logs the metrics from cross-validation results."""
        for idx, metric in enumerate(metrics):
            mlflow.log_metric(f"mean_{metric}", cv_results[idx].mean())
            mlflow.log_metric(f"std_{metric}", cv_results[idx].std())

    def run_mlflow_ui(self, port=5000, open_browser=True):
        """
        Starts the MLflow UI server on a specified port and optionally opens it in a web browser.
        """

        def is_server_running():
            """ Check if the MLflow UI server is already running """
            try:
                requests.get(f"http://localhost:{port}")
                return True
            except requests.ConnectionError:
                return False

        if not is_server_running():
            # Start the server if not already running
            def start_server():
                subprocess.run(["mlflow", "ui", "--port", str(port)])

            thread = threading.Thread(target=start_server)
            thread.daemon = True
            thread.start()

            if open_browser:
                time.sleep(2)  # Give the server a moment to start
                webbrowser.open(f"http://localhost:{port}")
