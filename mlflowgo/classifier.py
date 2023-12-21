from .artifact_logger import ArtifactLogger
from .tournament import Tournament
import mlflow
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score)


class Classifier(ArtifactLogger):
    """
    Classifier class for logging classification-specific artifacts and metrics to MLflow.

    This class extends the ArtifactLogger class to provide methods for logging classification-specific artifacts
    and metrics to MLflow during a machine learning experiment.

    Attributes:
        base (Tournament): The Tournament instance containing the experiment information.
    """
    def __init__(self, base: Tournament):
        """
        Initialize a Classifier instance.

        Args:
            base (Tournament): The Tournament instance containing the experiment information.

        Returns:
            None
        """
        super().__init__()
        self.base = base
        self.ignored_models_for_shap = [
            'NearestCentroid', 'AdaBoostClassifier', 'QuadraticDiscriminantAnalysis',
            'LinearSVC', 'LinearDiscriminantAnalysis', 'SVC', 'KNeighborsClassifier',
            'MLPClassifier', 'LabelPropagation', 'LabelSpreading']

    def log(self):
        """
        Log classification-specific artifacts and metrics to MLflow.

        This method logs classification-specific artifacts and metrics to MLflow, including ROC curve, precision-recall
        curve, calibration plot, confusion matrix, classification report, data sample, learning curve, validation curve,
        feature importance, SHAP plots, and experiment summary.

        Returns:
            None
        """
        y_pred = self.base.pipeline.predict(self.base.X_test)

        if hasattr(self.base.pipeline.named_steps[self.base.model_step], 'predict_proba'):
            y_scores = self.base.pipeline.predict_proba(self.base.X_test)
            # Log ROC curve
            self.log_roc_curve(self.base.y_test,
                               y_scores,
                               self.base.pipeline.named_steps[self.base.model_step].classes_)

            # Log precision recall curve
            self.log_precision_recall_curve(self.base.y_test,
                                            y_scores,
                                            self.base.pipeline.named_steps[self.base.model_step].classes_)
            
            # Log calibration plot
            self.log_calibration_plot(self.base.pipeline,
                                      self.base.X_test,
                                      self.base.y_test)

        # Log confusion matrix
        self.log_confusion_matrix(self.base.y_test,
                                  y_pred)

        # Log classification report
        self.log_classification_report(self.base.y_test,
                                       y_pred,
                                       self.base.pipeline.named_steps[self.base.model_step].classes_)

        # Log data sample
        self.log_data_sample(self.base.X_test,
                             10)  # Log 10 samples

        # Log learning curve
        self.log_learning_curves(self.base.pipeline,
                                 self.base.X_train,
                                 self.base.y_train,
                                 cv=5,
                                 scoring='f1_weighted')

        # Log validation curve
        if self.base.param_name is not None and self.base.param_range is not None:
            self.log_validation_curve(self.base.pipeline,
                                      self.base.X_train,
                                      self.base.y_train,
                                      param_name=f'{self.base.model_step}__{self.base.param_name}',
                                      param_range=self.base.param_range,
                                      cv=5,
                                      scoring='f1_weighted')

        # Log feature importance
        if hasattr(self.base.pipeline.named_steps[self.base.model_step], 'feature_importances_'):
            self.log_feature_importance(self.base.pipeline,
                                        self.base.model_step,
                                        self.base.feature_names)

        # Log SHAP
        if self.base.model_name not in self.ignored_models_for_shap:
            self.log_shap_summary_plot(self.base.pipeline,
                                       self.base.model_step,
                                       self.base.X_train)
            self.log_shap_partial_dependence_plot(self.base.pipeline,
                                                  self.base.model_step,
                                                  self.base.X_train)
            self.log_classification_shap_scatter_plot(self.base.pipeline,
                                                      self.base.model_step,
                                                      self.base.X_train)

        # Log exeriment summary
        self._generate_classification_experiment_summary()

    def _generate_classification_experiment_summary(self):
        """
        Generate a summary of the classification experiment.

        This method generates a summary of the classification experiment, including key findings and conclusions
        based on performance metrics.

        Returns:
            None
        """
        def analyse_classification_results(performance_metrics):
            """
            Analyze classification results and provide key findings.

            This function analyzes classification results based on performance metrics such as train accuracy and test accuracy.
            It provides key findings based on the comparison between these metrics, helping to identify potential issues
            like overfitting or exceptional performance.

            Args:
                performance_metrics (dict): A dictionary containing performance metrics, including 'Train Accuracy'
                    and 'Test Accuracy'.

            Returns:
                str: A string summarizing the key findings based on the analysis of performance metrics.
            """
            train_accuracy = performance_metrics['Train Accuracy']
            test_accuracy = performance_metrics['Test Accuracy']
            if test_accuracy < train_accuracy - 0.1:  # Threshold can be adjusted
                return "Model may be overfitting as test accuracy is significantly lower than train accuracy."
            elif test_accuracy > train_accuracy:
                return "Model performs exceptionally well on test data, which is rare and could indicate data leakage."
            else:
                return "Model generalises well from training to test data."

        def generate_classification_conclusions(performance_metrics):
            """
            Generate conclusions based on classification performance metrics.

            This function generates conclusions based on classification performance metrics such as precision and recall.
            It assesses whether the model struggles with precision and recall or exhibits a balanced performance.

            Args:
                performance_metrics (dict): A dictionary containing performance metrics, including 'Test Precision'
                    and 'Test Recall'.

            Returns:
                str: A string providing conclusions based on the analysis of classification performance metrics.
            """
            test_precision = performance_metrics['Test Precision']
            test_recall = performance_metrics['Test Recall']
            if test_precision < 0.6 or test_recall < 0.6:
                return "Model struggles with precision and/or recall. Consider improving feature selection or model tuning."
            else:
                return "Model shows balanced precision and recall. Explore potential for further optimisation."

        # Extract model type and hyperparameters
        model_step = self.base.pipeline.steps[-1][1]  # Assuming the model is the last step in the pipeline
        model_type = type(model_step).__name__
        hyperparameters = model_step.get_params()

        # Train the model and predict
        self.base.pipeline.fit(self.base.X_train, self.base.y_train)
        y_pred_train = self.base.pipeline.predict(self.base.X_train)
        y_pred_test = self.base.pipeline.predict(self.base.X_test)

        # Calculate performance metrics
        performance_metrics = {
            'Train Accuracy': accuracy_score(self.base.y_train, y_pred_train),
            'Test Accuracy': accuracy_score(self.base.y_test, y_pred_test),
            'Train Precision': precision_score(self.base.y_train, y_pred_train, average='weighted'),
            'Test Precision': precision_score(self.base.y_test, y_pred_test, average='weighted'),
            'Train Recall': recall_score(self.base.y_train, y_pred_train, average='weighted'),
            'Test Recall': recall_score(self.base.y_test, y_pred_test, average='weighted'),
            'Train F1-Score': f1_score(self.base.y_train, y_pred_train, average='weighted'),
            'Test F1-Score': f1_score(self.base.y_test, y_pred_test, average='weighted')
        }

        # Feature Importance (if applicable)
        feature_importance = ''
        if hasattr(model_step, 'feature_importances_'):
            importances = model_step.feature_importances_
            feature_importance = 'Feature Importances: ' + ', '.join([f'{feature}: {round(importance, 3)}' 
                                                                    for feature, importance in zip(self.base.X_train.columns, importances)])

        # Analyze results for key findings
        key_findings = analyse_classification_results(performance_metrics)

        # Generate dynamic conclusions based on metrics
        conclusions = generate_classification_conclusions(performance_metrics)

        # Construct summary
        hyperparameters_str = ', '.join([f'{k}: {v}' for k, v in hyperparameters.items()])
        performance_metrics_str = ', '.join([f'{k}: {v:.3f}' for k, v in performance_metrics.items()])

        description = (
            f"Experiment Overview:\n- Objective: {self.base.objective}\n- Model Type: {model_type}\n\n"
            f"Hyperparameters:\n- {hyperparameters_str}\n\n"
            f"Data Summary:\n- Dataset: {self.base.dataset_desc}\n\n"
            f"Performance Metrics:\n- {performance_metrics_str}\n\n"
            f"{feature_importance}\n\n"
            f"Key Findings:\n- {key_findings}\n\n"
            f"Conclusions:\n- {conclusions}"
        )

        # Update the MLflow experiment's description
        mlflow.set_tag("mlflow.note.content", description)
