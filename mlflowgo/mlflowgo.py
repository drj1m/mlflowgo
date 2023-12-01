from .artifact_logger import ArtifactLogger
from . import CLASSIFIER_KEY, REGRESSOR_KEY
from .base import Base
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier
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
        self.param_name = kwargs.get('param_name', None)
        self.param_range = kwargs.get('param_range', None)

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
            self._log_params(self.pipeline)
            if cv_results is not None: self._log_metrics(cv_results, self.metrics)
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
        y_pred = pipeline.predict(self.X_test)

        if self.task_type == CLASSIFIER_KEY:
            y_scores = pipeline.predict_proba(self.X_test)
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

            # Log calibration plot
            artifact_logger.log_calibration_plot(pipeline,
                                                 self.X_test,
                                                 self.y_test)
        elif self.task_type == REGRESSOR_KEY:
            # Log residual plot
            artifact_logger.log_residual_plot(pipeline,
                                              self.X_test,
                                              self.y_test)

            # Log predicted vs actual plot
            artifact_logger.log_prediction_vs_actual_plot(pipeline,
                                                          self.X_test,
                                                          self.y_test)

            # Log coefficient plot
            if hasattr(pipeline.named_steps[self.model_step], 'coef_'):
                artifact_logger.log_coefficient_plot(pipeline,
                                                     self.model_step,
                                                     self.feature_names)

            # Log regression report
            artifact_logger.log_regression_report(pipeline,
                                                  self.X_train,
                                                  self.y_train,
                                                  self.X_test,
                                                  self.y_test)

            # Log QQ plot
            artifact_logger.log_qq_plot(pipeline,
                                        self.X_test,
                                        self.y_test)

            # Log scale location plot
            artifact_logger.log_scale_location_plot(pipeline,
                                                    self.X_test,
                                                    self.y_test)

            # Log exeriment summary
            self._generate_regression_experiment_summary(pipeline,
                                                         self.X_train,
                                                         self.y_train,
                                                         self.X_test,
                                                         self.y_test)

        # Log data sample
        artifact_logger.log_data_sample(self.X_test,
                                        10)  # Log 10 samples

        # Log learning curve
        artifact_logger.log_learning_curves(pipeline,
                                            self.X_train,
                                            self.y_train,
                                            cv=5,
                                            scoring=self.metrics[0])

        # Log validation curve
        if self.param_name is not None and self.param_range is not None:
            artifact_logger.log_validation_curve(pipeline,
                                                 self.X_train,
                                                 self.y_train,
                                                 param_name=f'{self.model_step}__n_estimators',
                                                 param_range=[50, 100, 200, 500],
                                                 cv=5,
                                                 scoring=self.metrics[0])

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
        for idx, metric in enumerate(metrics):
            mlflow.log_metric(f"mean_{metric}", cv_results[idx].mean())
            mlflow.log_metric(f"std_{metric}", cv_results[idx].std())

    def _generate_regression_experiment_summary(self, pipeline, X_train, y_train, X_test, y_test, objective='', dataset_desc=''):
        """
        Generates and updates an MLflow experiment summary based on a training pipeline.

        Parameters:
        pipeline (sklearn.pipeline.Pipeline): Object type that implements the "fit" and "predict" methods
        X_train, y_train (pd.DataFrame): Training dataset (features and target).
        X_test, y_test (pd.DataFrame): Test dataset (features and target).
        objective (str): The objective of the experiment.
        dataset_desc (str): Description of the dataset used.
        """
        def analyze_results(performance_metrics):
            """
            Analyze performance metrics to generate key findings.
            """
            # Example analysis logic
            train_rmse = performance_metrics['Train RMSE']
            test_rmse = performance_metrics['Test RMSE']
            if test_rmse > train_rmse * 1.2:
                return "Model may be overfitting as test RMSE is significantly higher than train RMSE."
            elif test_rmse < train_rmse:
                return "Model performs better on test set, which is unusual and may suggest data leakage or overfitting."
            else:
                return "Model generalizes well from training to test data."

        def generate_conclusions(performance_metrics):
            """
            Generate dynamic conclusions based on performance metrics.
            """
            # Example logic for conclusion
            test_r2 = performance_metrics['Test R2']
            if test_r2 > 0.8:
                return "Model shows high predictive accuracy on test data. Further tuning may focus on feature selection."
            elif test_r2 < 0.5:
                return "Model underperforms on test data. Consider revising model complexity or feature engineering."
            else:
                return "Model shows moderate performance. Further improvements can be made in model tuning."

        # Extract model type and hyperparameters
        model_step = pipeline.steps[-1][1]  # Assuming the model is the last step in the pipeline
        model_type = type(model_step).__name__
        hyperparameters = model_step.get_params()

        # Train the model and predict
        pipeline.fit(X_train, y_train)
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        # Calculate performance metrics
        performance_metrics = {
            'Train RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'Train R2': r2_score(y_train, y_pred_train),
            'Test R2': r2_score(y_test, y_pred_test)
        }

        # Analyze results for key findings
        key_findings = analyze_results(performance_metrics)

        # Generate dynamic conclusions based on metrics
        conclusions = generate_conclusions(performance_metrics)

        # Construct summary
        hyperparameters_str = ', '.join([f'{k}: {v}' for k, v in hyperparameters.items()])
        performance_metrics_str = ', '.join([f'{k}: {v:.3f}' for k, v in performance_metrics.items()])

        description = (
            f"**Experiment Overview:**\n- Objective: {objective}\n- Model Type: {model_type}\n\n"
            f"**Hyperparameters:**\n- {hyperparameters_str}\n\n"
            f"**Data Summary:**\n- Dataset: {dataset_desc}\n\n"
            f"**Performance Metrics:**\n- {performance_metrics_str}\n\n"
            f"**Key Findings:**\n- {key_findings}\n\n"
            f"**Conclusions:**\n- {conclusions}"
        )

        # Update the MLflow experiment's description
        mlflow.set_tag("mlflow.note.content", description)

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
