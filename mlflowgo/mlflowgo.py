from .artifact_logger import ArtifactLogger
from . import CLASSIFIER_KEY
from .base import Base
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier
import subprocess
import threading
import webbrowser
import requests
import time
import pandas as pd


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

    def run_experiment(self, X: pd.DataFrame, y: pd.DataFrame, cv: int = 5, **kwargs):
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
        self.pipeline, self.task_type = self.get_pipeline(
            kwargs.get('pipeline', None),
            kwargs.get('task_type', None))
        self.metrics = self.get_model_metrics(
            kwargs.get('metrics', None),
            self.task_type)
        self.feature_names = self.get_feature_names(
            kwargs.get('feature_names', None),
            X.columns
        )
        self.model_step = self.get_model_step_from_pipeline(self.pipeline)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.33)

        with mlflow.start_run(run_name=self.get_run_name(self.pipeline)):
            # Perform cross-validation
            cv_results = cross_val_score(
                self.pipeline,
                self.X_train,
                self.y_train,
                cv=cv,
                scoring=self.metrics[0])

            # Log parameters, metrics, and model
            self._log_params(self.pipeline)
            self._log_metrics(cv_results, self.metrics)
            if self.task_type == CLASSIFIER_KEY:
                self._log_artifacts(self.pipeline,
                                    self.feature_names)
            mlflow.sklearn.log_model(self.pipeline,
                                     self.model_step)

    def _log_artifacts(self, pipeline, feature_names):
        """
        Log all relevant artifacts for the experiment

        Parameters:
            pipeline (sklearn.pipeline.Pipeline): The scikit-learn compatible pipeline to evaluate.
            feature_names (array-like): Typically the column names of X
        """
        artifact_logger = ArtifactLogger()
        pipeline.fit(self.X_train, self.y_train)
        y_pred, y_scores = pipeline.predict(self.X_test), pipeline.predict_proba(self.X_test)

        if is_classifier(pipeline):
            # Log ROC curve
            artifact_logger.log_roc_curve(self.y_test,
                                          y_scores,
                                          pipeline.named_steps[self.model_step].classes_)
            # Log confusion matrix
            artifact_logger.log_confusion_matrix(self.y_test,
                                                 y_pred)

            # Log precision recall curve
            artifact_logger.log_precision_recall_curve(self.y_test,
                                                       y_scores,
                                                       pipeline.named_steps[self.model_step].classes_)

            # Log classification report
            artifact_logger.log_classification_report(self.y_test,
                                                      y_pred,
                                                      pipeline.named_steps[self.model_step].classes_)
            # Log learning curve
            artifact_logger.log_learning_curves(pipeline,
                                                self.X_train,
                                                self.y_train,
                                                cv=5,
                                                scoring=self.metrics[0])

            # Log data sample
            artifact_logger.log_data_sample(self.X_test,
                                            10)  # Log 10 samples

        # Log feature importance
        if hasattr(pipeline.named_steps[self.model_step], 'feature_importances_'):
            artifact_logger.log_feature_importance(pipeline,
                                                   self.model_step,
                                                   feature_names)

    def _log_params(self, pipeline):
        """
        Logs the parameters of the model or pipeline.

        Parameters:
            pipeline (sklearn.pipeline.Pipeline): The scikit-learn compatible pipeline to evaluate.
        """
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
