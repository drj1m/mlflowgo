from .regressor import Regressor
from .classifier import Classifier
from . import CLASSIFIER_KEY, REGRESSOR_KEY
from .tournament import Tournament
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics as sklm
import subprocess
import threading
import webbrowser
import requests
import time
import pandas as pd


class MLFlowGo():
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

    def run_experiment(self, X: pd.DataFrame, y: pd.DataFrame, cv: int = 5, pipeline: Pipeline = None, grid_search=False, register=False, **kwargs):
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

        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33)

        tournament = Tournament(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            pipelines=pipeline
        )

        for _pipeline in tournament.pipelines:

            with mlflow.start_run(run_name=tournament.get_run_name(_pipeline)) as run:

                tournament.run(
                    run_id=run.info.run_id,
                    pipeline=_pipeline,
                    grid_search=False,
                    cv=-1)

                # Log parameters, metrics, and model
                self._log_params(tournament.pipeline)
                self._log_metrics(
                    tournament.y_test,
                    tournament.pipeline.predict(tournament.X_test),
                    tournament.metrics,
                    tournament.task_type
                )
                self._log_artifacts(tournament)
                mlflow.sklearn.log_model(tournament.pipeline,
                                         tournament.model_name)

        best_model_name = min(tournament.final_scores, key=tournament.final_scores.get)
        print(f"Optimal model: {best_model_name}")
        if register:
            model_uri = f"runs:/{tournament.model_info[best_model_name][0]}/{tournament.model_info[best_model_name][1]}"
            mlflow.register_model(model_uri, best_model_name)

    def _log_artifacts(self, tournament):
        """
        Log all relevant artifacts for the experiment
        """

        if tournament.task_type == REGRESSOR_KEY:
            artifact_logger = Regressor(tournament)
        elif tournament.task_type == CLASSIFIER_KEY:
            artifact_logger = Classifier(tournament)

        artifact_logger.log()

    def _log_params(self, pipeline):
        """
        Logs the parameters of the model or pipeline.
        """
        if isinstance(pipeline, Pipeline):
            params = pipeline.get_params()
        else:
            params = pipeline.__dict__
        mlflow.log_params(params)

    def _log_metrics(self, y_true, y_pred, metrics, task_type):
        """Logs the metrics from cross-validation results."""
        if task_type == REGRESSOR_KEY:
            metric_func = {
                'max_error': lambda y_true, y_pred: sklm.max_error(y_true, y_pred),
                'neg_median_absolute_error': lambda y_true, y_pred: sklm.median_absolute_error(y_true, y_pred),
                'neg_mean_absolute_error': lambda y_true, y_pred: sklm.mean_absolute_error(y_true, y_pred),
                'neg_mean_absolute_percentage_error': lambda y_true, y_pred: sklm.mean_absolute_percentage_error(y_true, y_pred),
                'neg_mean_squared_error': lambda y_true, y_pred: sklm.mean_squared_error(y_true, y_pred),
                'neg_mean_squared_log_error': lambda y_true, y_pred: sklm.mean_squared_log_error(y_true, y_pred),
                'neg_root_mean_squared_error': lambda y_true, y_pred: sklm.mean_squared_error(y_true, y_pred, squared=False)
            }
        elif task_type == CLASSIFIER_KEY:
            metric_func = {
                'accuracy': lambda y_true, y_pred: sklm.accuracy_score(y_true, y_pred),
                'precision': lambda y_true, y_pred: sklm.precision_score(y_true, y_pred, average='weighted'),
                'recall': lambda y_true, y_pred: sklm.recall_score(y_true, y_pred, average='weighted'),
                'f1_score': lambda y_true, y_pred: sklm.f1_score(y_true, y_pred, average='weighted'),
                'roc_auc_score': lambda y_true, y_pred: sklm.roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr'),
                'balanced_accuracy': lambda y_true, y_pred: sklm.balanced_accuracy_score(y_true, y_pred),
                'jaccard_score': lambda y_true, y_pred: sklm.jaccard_score(y_true, y_pred, average='weighted'),
                'log_loss': lambda y_true, y_pred: sklm.log_loss(y_true, y_pred),
                'precision_recall_fscore_support': lambda y_true, y_pred: sklm.precision_recall_fscore_support(y_true, y_pred, average='weighted')
            }

        for metric_name in metrics:
            if metric_name in metric_func:
                result = metric_func[metric_name](y_true, y_pred)
                mlflow.log_metric(f"{metric_name}", result)

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
