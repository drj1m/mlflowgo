from .artifact_logger import ArtifactLogger
from .tournament import Tournament
import mlflow
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


class Regressor(ArtifactLogger):
    """ A class to handle logging artifacts to MLFlow for regression model
    """

    def __init__(self, base: Tournament):
        super().__init__()
        self.base = base

    def log(self):
        """ Log artifacts to MLFlow
        """

        # Log residual plot
        self.log_residual_plot(self.base.pipeline,
                               self.base.X_test,
                               self.base.y_test)

        # Log predicted vs actual plot
        self.log_prediction_vs_actual_plot(self.base.pipeline,
                                           self.base.X_test,
                                           self.base.y_test)

        # Log coefficient plot
        if hasattr(self.base.pipeline.named_steps[self.base.model_step], 'coef_'):
            self.log_coefficient_plot(self.base.pipeline,
                                      self.base.model_step,
                                      self.base.feature_names)

        # Log regression report
        self.log_regression_report(self.base.pipeline,
                                   self.base.X_train,
                                   self.base.y_train,
                                   self.base.X_test,
                                   self.base.y_test)

        # Log QQ plot
        self.log_qq_plot(self.base.pipeline,
                         self.base.X_test,
                         self.base.y_test)

        # Log scale location plot
        self.log_scale_location_plot(self.base.pipeline,
                                     self.base.X_test,
                                     self.base.y_test)

        # Log exeriment summary
        self._generate_regression_experiment_summary()

        # Log data sample
        self.log_data_sample(self.base.X_test,
                             10)  # Log 10 samples

        # Log learning curve
        self.log_learning_curves(self.base.pipeline,
                                 self.base.X_train,
                                 self.base.y_train,
                                 cv=5,
                                 scoring='neg_mean_squared_error')

        # Log validation curve
        if self.base.param_name is not None and self.base.param_range is not None:
            self.log_validation_curve(self.base.pipeline,
                                      self.base.X_train,
                                      self.base.y_train,
                                      param_name=f'{self.base.model_step}__{self.base.param_name}',
                                      param_range=self.base.param_range,
                                      cv=5,
                                      scoring='neg_mean_squared_error')

        # Log feature importance
        if hasattr(self.base.pipeline.named_steps[self.base.model_step], 'feature_importances_'):
            self.log_feature_importance(self.base.pipeline,
                                        self.base.model_step,
                                        self.base.feature_names)

        # Log SHAP
        if not isinstance(self.base.pipeline.named_steps[self.base.model_step],
                          (SVC, SVR, MLPClassifier, MLPRegressor)):
            self.log_shap_summary_plot(self.base.pipeline,
                                       self.base.model_step,
                                       self.base.X_train)
            self.log_shap_partial_dependence_plot(self.base.pipeline,
                                                  self.base.model_step,
                                                  self.base.X_train)
            self.log_regression_shap_scatter_plot(self.base.pipeline,
                                                  self.base.model_step,
                                                  self.base.X_train)

    def _generate_regression_experiment_summary(self):
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
        model_step = self.base.pipeline.steps[-1][1]  # Assuming the model is the last step in the pipeline
        model_type = type(model_step).__name__
        hyperparameters = model_step.get_params()

        # Train the model and predict
        self.base.pipeline.fit(self.base.X_train, self.base.y_train)
        y_pred_train = np.nan_to_num(self.base.pipeline.predict(self.base.X_train))
        y_pred_test = np.nan_to_num(self.base.pipeline.predict(self.base.X_test))

        # Calculate performance metrics
        performance_metrics = {
            'Train RMSE': np.sqrt(mean_squared_error(self.base.y_train, y_pred_train)),
            'Test RMSE': np.sqrt(mean_squared_error(self.base.y_test, y_pred_test)),
            'Train R2': r2_score(self.base.y_train, y_pred_train),
            'Test R2': r2_score(self.base.y_test, y_pred_test)
        }

        # Analyze results for key findings
        key_findings = analyze_results(performance_metrics)

        # Generate dynamic conclusions based on metrics
        conclusions = generate_conclusions(performance_metrics)

        # Construct summary
        hyperparameters_str = ', '.join([f'{k}: {v}' for k, v in hyperparameters.items()])
        performance_metrics_str = ', '.join([f'{k}: {v:.3f}' for k, v in performance_metrics.items()])

        description = (
            f"**Experiment Overview:**\n- Objective: {self.base.objective}\n- Model Type: {model_type}\n\n"
            f"**Hyperparameters:**\n- {hyperparameters_str}\n\n"
            f"**Data Summary:**\n- Dataset: {self.base.dataset_desc}\n\n"
            f"**Performance Metrics:**\n- {performance_metrics_str}\n\n"
            f"**Key Findings:**\n- {key_findings}\n\n"
            f"**Conclusions:**\n- {conclusions}"
        )

        # Update the MLflow experiment's description
        mlflow.set_tag("mlflow.note.content", description)
