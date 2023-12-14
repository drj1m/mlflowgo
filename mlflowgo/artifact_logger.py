from .artifact_base import ArtifactBase
import mlflow
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    classification_report, mean_squared_error, mean_absolute_error, r2_score)
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.calibration import calibration_curve
import shap
import scipy.stats as stats
import numpy as np
import pandas as pd
import os
import tempfile


class ArtifactLogger:
    """
    A class to create and log artifacts.
    """

    def __init__(self):
        pass

    def log_learning_curves(self, pipeline, X, y, cv, scoring):
        """
        Generate and log learning curve plots as MLflow artifacts.

        This function generates learning curve plots for a given scikit-learn pipeline and logs them as MLflow artifacts.

        Parameters:
        pipeline (sklearn.pipeline.Pipeline): Scikit-learn pipeline object implementing "fit" and "predict" methods.
        X (pd.DataFrame): Feature dataset.
        y (pd.DataFrame): Target values.
        cv (int): Number of cross-validation splits.
        scoring (str or callable): A scoring metric for evaluation. Refer to scikit-learn's model evaluation documentation.

        Notes:
        - The learning curves are computed using cross-validation.
        - Learning curves provide insights into model performance as training data size increases.
        - The generated plots show the training score and cross-validation score with varying training dataset sizes.

        Example:
        ```python
        classifier = Classifier(base=tournament)
        classifier.log_learning_curves(pipeline, X_train, y_train, cv=5, scoring='f1_weighted')
        ```

        """
        train_sizes, train_scores, test_scores = learning_curve(
            pipeline,
            X,
            y,
            cv=cv,
            n_jobs=-1,
            scoring=scoring,
            train_sizes=np.linspace(0.1, 1.0, 10))

        # Calculate mean and standard deviation for training set scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # Calculate mean and standard deviation for test set scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot learning curves
        plt.plot(train_sizes, train_mean, label="Training score", color="blue", marker='o')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.15)

        plt.plot(train_sizes, test_mean, label="Cross-validation score", color="green", marker='o')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="green", alpha=0.15)

        plt.title("Learning Curves")
        plt.xlabel("Training Data Size")
        plt.ylabel(scoring)
        plt.legend(loc="best")

        # Save the plot to a temporary file and log it
        with tempfile.NamedTemporaryFile(suffix="__learning curves.png", delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()

            # Log the temporary file as an artifact
            mlflow.log_artifact(tmp.name, 'Model Selection')

        # Remove the temporary file
        os.remove(tmp.name)

    def log_validation_curve(self, pipeline, X, y, param_name, param_range, cv, scoring):
        """
        Generates and logs validation curve plot as an MLflow artifact.

        Parameters:
        pipeline (sklearn.pipeline.Pipeline): object type that implements the "fit" and "predict" methods
        X (pd.DataFrame): Feature dataset.
        y (pd.DataFrame): Target values.
        param_name(str): Name of the parameter to vary.
        param_range (array-like): The values of the parameter that will be evaluated.
        cv (int, optional): Number of cross-validation splits.
        scoring(str): A str (see model evaluation documentation) or a scorer callable object/function.
        """
        train_scores, test_scores = validation_curve(
            pipeline,
            X,
            y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        # Calculate mean and standard deviation for training set scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # Calculate mean and standard deviation for test set scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot validation curves
        plt.plot(param_range, train_mean, label="Training score", color="blue", marker='o')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.15)

        plt.plot(param_range, test_mean, label="Cross-validation score", color="green", marker='o')
        plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="green", alpha=0.15)

        plt.title("Validation Curve")
        plt.xlabel(param_name)
        plt.ylabel(scoring)
        plt.legend(loc="best")

        # Save the plot to a temporary file and log it
        with tempfile.NamedTemporaryFile(suffix="__validation curve.png", delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()

            # Log the temporary file as an artifact
            mlflow.log_artifact(tmp.name, 'Model Selection')

        # Remove the temporary file
        os.remove(tmp.name)

    def log_calibration_plot(self, pipeline, X, y, n_bins=10, strategy='uniform'):
        """
        Determine which function to call to produce a calibration plot.

        Parameters:
        pipeline (sklearn.pipeline.Pipeline): object type that implements the "fit" and "predict" methods
        X (pd.DataFrame): Feature dataset.
        y (pd.DataFrame): Target values.
        n_bins(int default=10): The number of bins to use for calibration.
        strategy (str {'uniform', 'quantile'}, default='uniform'): Strategy used to define the widths of the bins.
        """
        if len(np.unique(y)) > 2:
            self._log_calibration_plot_one_vs_rest(pipeline, X, y, n_bins=10, strategy='uniform')
        else:
            self._log_binary_calibration_plot(pipeline, X, y, n_bins=10, strategy='uniform')

    def _log_calibration_plot_one_vs_rest(self, pipeline, X, y, n_bins=10, strategy='uniform'):
        """
        Generates and logs a one-vs-rest calibration plot as an MLflow artifact for multi-class data.

        Parameters:
        pipeline (sklearn.pipeline.Pipeline): object type that implements the "fit" and "predict" methods
        X (pd.DataFrame): Feature dataset.
        y (pd.DataFrame): Target values.
        n_bins(int default=10): The number of bins to use for calibration.
        strategy (str {'uniform', 'quantile'}, default='uniform'): Strategy used to define the widths of the bins.
        """
        classes = np.unique(y)
        plt.figure(figsize=(8, 6))

        y_proba = pipeline.predict_proba(X)

        for idx, cls in enumerate(classes):
            y_class = (y == cls).astype(int)  # One-vs-rest for current class
            class_proba = y_proba[:, idx]

            prob_true, prob_pred = calibration_curve(y_class, class_proba, n_bins=n_bins, strategy=strategy)
            plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=f'Class {cls}')

        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        plt.xlabel('Predicted probability')
        plt.ylabel('True probability in each bin')
        plt.legend()
        plt.title('One-vs-Rest Calibration Plot')

        # Save the plot to a temporary file and log it
        with tempfile.NamedTemporaryFile(suffix="__one vs rest.png", delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()

            # Log the temporary file as an artifact
            mlflow.log_artifact(tmp.name, 'Calibration')

        # Remove the temporary file
        os.remove(tmp.name)

    def _log_binary_calibration_plot(self, pipeline, X, y, n_bins=10, strategy='uniform'):
        """
        Generates and logs a calibration plot as an MLflow artifact.

        Parameters:
        pipeline (sklearn.pipeline.Pipeline): object type that implements the "fit" and "predict" methods
        X (pd.DataFrame): Feature dataset.
        y (pd.DataFrame): Target values.
        n_bins(int default=10): The number of bins to use for calibration.
        strategy (str {'uniform', 'quantile'}, default='uniform'): Strategy used to define the widths of the bins.
        """
        # Predict probabilities
        y_proba = pipeline.predict_proba(X)[:, 1]

        # Calibration curve
        prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=n_bins, strategy=strategy)

        # Plot calibration curve
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration plot')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        plt.xlabel('Predicted probability')
        plt.ylabel('True probability in each bin')
        plt.legend()
        plt.title('Calibration Plot')

        # Save the plot to a temporary file and log it
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()

            # Log the temporary file as an artifact
            mlflow.log_artifact(tmp.name, 'Calibration')

        # Remove the temporary file
        os.remove(tmp.name)

    def log_roc_curve(self, y_true, y_scores, class_names):
        """
        Determine which function to call to produce a ROC curve.

        Parameters:
        y_true (array-like): True labels of the data.

        y_scores (array-like): Target scores. Can either be probability estimates, confidence values, 
                or binary decisions.

        class_names (list): List of class names.
        """
        if len(np.unique(y_true)) > 2:
            self._log_multi_class_roc_curve(y_true, y_scores, class_names)
        else:
            self._log_binary_roc_curve(y_true, y_scores)

    def _log_binary_roc_curve(self, y_true, y_scores):
        """
        Logs a ROC curve for a binary classifier as an MLflow artifact.

        Parameters:
        y_true (array-like): True labels of the data.

        y_scores (array-like): Target scores. Can either be probability estimates, confidence values, 
                or binary decisions.
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        with tempfile.NamedTemporaryFile(suffix="__roc.png", delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()
            mlflow.log_artifact(tmp.name, 'Metrics')
        os.remove(tmp.name)

    def _log_multi_class_roc_curve(self, y_true, y_scores, class_names):
        """
        Logs a ROC curve for a multi-class classifier as an MLflow artifact.

        Parameters:
        y_true (array-like): True labels of the data.

        y_scores (array-like): Target scores. Can either be probability estimates, confidence values, 
                or binary decisions.

        class_names (list): List of class names corresponding to the labels.
        """

        # Binarize the output for multi-class
        y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))

        # Compute ROC curve and ROC area for each class
        n_classes = y_true_binarized.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot the ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
                 color='navy', linestyle=':', linewidth=4)

        for i, color in zip(range(n_classes), plt.cm.rainbow(np.linspace(0, 1, n_classes))):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        with tempfile.NamedTemporaryFile(suffix="__roc.png", delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()
            mlflow.log_artifact(tmp.name, 'Metrics')
        os.remove(tmp.name)

    def log_precision_recall_curve(selk, y_true, y_scores, class_names):
        """
        Logs a precision-recall curve plot as an MLflow artifact.

        Parameters:
        y_true (array-like): True labels of the data.

        y_scores (array-like): Target scores. Can either be probability estimates, confidence values, 
                or binary decisions.

        class_names (list): List of class names corresponding to the labels.
        """
        # Compute Precision-Recall and plot curve
        precision = dict()
        recall = dict()
        average_precision = dict()
        n_classes = len(class_names)

        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true == i, y_scores[:, i])
            average_precision[i] = average_precision_score(y_true == i, y_scores[:, i])

        # Plot the Precision-Recall curve for each class
        plt.figure(figsize=(7, 7))
        for i, color in zip(range(n_classes), plt.cm.viridis(np.linspace(0, 1, n_classes))):
            plt.plot(recall[i], precision[i], color=color, lw=2,
                    label=f'Precision-Recall curve of class {class_names[i]} (area = {average_precision[i]:0.2f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")

        # Save the plot to a temporary file and log it
        with tempfile.NamedTemporaryFile(suffix="__precision_recall.png", delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()

            # Log the temporary file as an artifact
            mlflow.log_artifact(tmp.name, 'Metrics')

        # Remove the temporary file
        os.remove(tmp.name)

    def log_feature_importance(self, pipeline, model_step, feature_names):
        """
        Logs basic feature importance as an MLflow artifact.

        Parameters:
        pipeline (sklearn.Pipeline): pipeline object.

        model_step (string): Step name for the model in the pipeline.

        feature_names (list): List of feature names.
        """
        importances = pipeline.named_steps[model_step].feature_importances_
        indices = np.argsort(importances)

        plt.figure()
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        with tempfile.NamedTemporaryFile(suffix="__feature_importance.png", delete=False) as tmp:
            plt.savefig(tmp.name, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(tmp.name, 'Metrics')
        os.remove(tmp.name)

    def log_shap_summary_plot(self, pipeline, model_step, X):
        """
        Generates and logs a SHAP summary plot to MLflow.

        Parameters:
        pipeline: Reference to the sklearn pipeline object.
        model_step (string): Step name for the model in the pipeline.
        X (pd.DataFrame): The input features used for prediction and SHAP value calculation.
        """
        explainer, X = ArtifactBase.get_shap_explainer(pipeline, model_step, X)

        # Calculate SHAP values
        if isinstance(explainer, shap.ExactExplainer):
            shap_values = explainer(X)
        else:
            shap_values = explainer.shap_values(X)

        # SHAP Summary Plot
        if hasattr(pipeline.named_steps[model_step], 'classes_') and len(shap_values) == len(pipeline.named_steps[model_step].classes_):
            for idx, _class in enumerate(pipeline.named_steps[model_step].classes_):
                shap.summary_plot(shap_values[idx], X, show=False)
                with tempfile.NamedTemporaryFile(suffix=f"__class {_class}.png", delete=False) as tmp:
                    plt.title(f"SHAP summary plot for class: {_class}")
                    plt.savefig(tmp.name, bbox_inches="tight")
                    plt.close()
                    mlflow.log_artifact(tmp.name, "SHAP/Summary Plot")
                    os.remove(tmp.name)
        else:
            shap.summary_plot(shap_values, X, show=False, title="SHAP summary plot")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                plt.savefig(tmp.name, bbox_inches="tight")
                plt.close()
                mlflow.log_artifact(tmp.name, "SHAP/Summary Plot")
                os.remove(tmp.name)

    def log_shap_partial_dependence_plot(self, pipeline, model_step, X):
        """
        Generates and logs partial dependency plots for each feature to MLflow

        Parameters:
        pipeline: Reference to the sklearn pipeline object.
        model_step (string): Step name for the model in the pipeline.
        X (pd.DataFrame): The input features used for prediction and SHAP value calculation.
        """

        _, X = ArtifactBase.get_shap_explainer(pipeline, model_step, X)

        for feature in X.columns:
            shap.partial_dependence_plot(
                feature,
                pipeline.named_steps[model_step].predict,
                X,
                ice=False,
                model_expected_value=True,
                feature_expected_value=True,
                show=False
            )
            with tempfile.NamedTemporaryFile(suffix=f"__{feature}.png", delete=False) as tmp:
                plt.savefig(tmp.name, bbox_inches="tight")
                plt.close()
                mlflow.log_artifact(tmp.name, "SHAP/Partial Dependence Plot")
                os.remove(tmp.name)

    def log_regression_shap_scatter_plot(self, pipeline, model_step, X):
        """
        Generates and logs scatter plots for each feature in MLflow

        Parameters:
        pipeline: Reference to the sklearn pipeline object.
        model_step (string): Step name for the model in the pipeline.
        X (pd.DataFrame): The input features used for predictions and SHAP value calculation.
        """

        explainer, X = ArtifactBase.get_shap_explainer(pipeline, model_step, X)
        shap_values = explainer(X)
        for idx in range(X.shape[1]):
            try:
                shap.plots.scatter(shap_values[:, idx], show=False)
                with tempfile.NamedTemporaryFile(suffix=f"__{X.columns.values[idx]}.png", delete=False) as tmp:
                    plt.savefig(tmp.name, bbox_inches="tight")
                    plt.close()
                    mlflow.log_artifact(tmp.name, "SHAP/Scatter Plot")
                    os.remove(tmp.name)
            except IndexError:
                continue

    def log_classification_shap_scatter_plot(self, pipeline, model_step, X):
        """
        Generates and logs scatter plots for each feature in MLflow

        Parameters:
        pipeline: Reference to the sklearn pipeline object.
        model_step (string): Step name for the model in the pipeline.
        X (pd.DataFrame): The input features used for predictions and SHAP value calculation.
        """
        model = pipeline.named_steps[model_step]
        explainer, X = ArtifactBase.get_shap_explainer(pipeline, model_step, X)
        shap_values = explainer(X)
        for class_idx in range(len(model.classes_)):
            for idx in range(X.shape[1]): 
                try:
                    shap.plots.scatter(shap_values[:, idx][:, class_idx], show=False)
                    with tempfile.NamedTemporaryFile(suffix=f"__{X.columns.values[idx]}_class {model.classes_[class_idx]}.png", delete=False) as tmp:
                        plt.savefig(tmp.name, bbox_inches="tight")
                        plt.close()
                        mlflow.log_artifact(tmp.name, f"SHAP/Scatter Plot")
                        os.remove(tmp.name)
                except IndexError:
                    shap.plots.scatter(shap_values[:, idx], show=False)
                    with tempfile.NamedTemporaryFile(suffix=f"__{X.columns.values[idx]}.png", delete=False) as tmp:
                        plt.savefig(tmp.name, bbox_inches="tight")
                        plt.close()
                        mlflow.log_artifact(tmp.name, f"SHAP/Scatter Plot")
                        os.remove(tmp.name)

    def log_confusion_matrix(self, y_true, y_pred):
        """
        Logs confusion matrix as an MLflow artifact.

        Parameters:
        y_true (array-like): The true values for y.

        y_pred (array-like): The prediction values for y.
        """
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots()
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.grid(False)
        with tempfile.NamedTemporaryFile(suffix="__confusion_matrix.png", delete=False) as tmp:
            plt.savefig(tmp.name, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(tmp.name, 'Metrics')
        os.remove(tmp.name)

    def log_data_sample(self, data, sample_size):
        """
        Logs a data sample as an MLflow artifact.

        Parameters:
        data (array-like): The feature dataset.

        sample_size (int): The size of the sample to store.
        """
        sample = pd.DataFrame(data).sample(n=sample_size)
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as tmp:
            sample.to_csv(tmp.name, index=False)
            mlflow.log_artifact(tmp.name, 'Data Sample')
        os.remove(tmp.name)

    def log_classification_report(self, y_true, y_pred, class_names):
        """
        Logs a classification report as an MLflow artifact.

        Parameters:
        y_true (array-like): True labels of the data.

        y_pred (array-like): Predicted labels of the data.

        class_names (list): List of class names corresponding to the labels.
        """
        if not isinstance(class_names[0], str):
            class_names = [str(i) for i in class_names]
        # Generate the classification report
        report = classification_report(y_true, y_pred, target_names=class_names)

        # Create a temporary file to save the report
        with tempfile.NamedTemporaryFile(mode='w', suffix="__classification_report.txt", delete=False) as tmp:
            tmp.write(report)
            tmp.flush()

            # Log the temporary file as an artifact
            mlflow.log_artifact(tmp.name, 'Metrics')

        # Remove the temporary file
        os.remove(tmp.name)

    def log_residual_plot(self, pipeline, X, y):
        """
        Generates and logs a residual plot as an MLflow artifact.

        Parameters:
        pipeline (sklearn.pipeline.Pipeline): object type that implements the "fit" and "predict" methods
        X (pd.DataFrame): Feature dataset.
        y (pd.DataFrame): Target values.
        """
        # Predict the values using the model
        y_pred = pipeline.predict(X)

        # Calculate residuals
        residuals = y - y_pred

        # Plotting the residuals
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, color='blue', s=10)
        plt.axhline(y=0, color='black', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')

        # Save the plot to a temporary file and log it
        with tempfile.NamedTemporaryFile(suffix="__residual.png", delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()

            # Log the temporary file as an artifact
            mlflow.log_artifact(tmp.name, 'Metrics')

        # Remove the temporary file
        os.remove(tmp.name)

    def log_prediction_vs_actual_plot(self, pipeline, X, y):
        """
        Generates and logs a prediction vs. actual plot as an MLflow artifact.

        Parameters:
        pipeline (sklearn.pipeline.Pipeline): object type that implements the "fit" and "predict" methods
        X (pd.DataFrame): Feature dataset.
        y (pd.DataFrame): Target values.
        """
        # Predict the values using the model
        y_pred = pipeline.predict(X)

        # Plotting prediction vs actual values
        plt.figure(figsize=(8, 6))
        plt.scatter(y, y_pred, color='blue', s=10)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Diagonal line
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Prediction vs. Actual')

        # Save the plot to a temporary file and log it
        with tempfile.NamedTemporaryFile(suffix="__prediction vs actual.png", delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()

            # Log the temporary file as an artifact
            mlflow.log_artifact(tmp.name, 'Metrics')

        # Remove the temporary file
        os.remove(tmp.name)

    def log_coefficient_plot(self, pipeline, model_step, feature_names):
        """
        Generates and logs a coefficient plot as an MLflow artifact for linear regression models
        or others with a 'coef_' attribute.

        Parameters:
        pipeline (sklearn.pipeline.Pipeline): Object type that implements the "fit" and "predict" methods
        model_step (str): Step name for the model
        feature_names (list): A list of names for the features corresponding to the coefficients.
        """
        # Ensure the model has the attribute 'coef_'
        if not hasattr(pipeline.named_steps[model_step], 'coef_'):
            raise ValueError("The provided estimator does not have 'coef_' attribute.")

        # Ensure the number of feature names matches the number of coefficients
        if len(feature_names) != len(pipeline.named_steps[model_step].coef_):
            raise ValueError("The number of feature names must match the number of coefficients.")

        # Plotting the coefficients
        plt.figure(figsize=(10, 6))
        plt.bar(feature_names, pipeline.named_steps[model_step].coef_)
        plt.xlabel('Features')
        plt.ylabel('Coefficient Value')
        plt.xticks(rotation=45, ha='right')
        plt.title('Feature Coefficients')

        # Save the plot to a temporary file and log it
        with tempfile.NamedTemporaryFile(suffix="__coefficient.png", delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()

            # Log the temporary file as an artifact
            mlflow.log_artifact(tmp.name, 'Metrics')

        # Remove the temporary file
        os.remove(tmp.name)

    def log_regression_report(self, pipeline, X_train, y_train, X_test, y_test):
        """
        Generates and logs a model summary as an MLflow artifact for regression models.

        Parameters:
        pipeline (sklearn.pipeline.Pipeline): Object type that implements the "fit" and "predict" methods
        X_train, y_train (pd.DataFrame): Training dataset (features and target).
        X_test, y_test (pd.DataFrame): Test dataset (features and target).
        """
        # Predictions on training and test sets
        y_train_pred = np.nan_to_num(pipeline.predict(X_train))
        y_test_pred = np.nan_to_num(pipeline.predict(X_test))

        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Prepare summary text
        summary_text = (
            f"Model Summary:\n"
            f"Training MSE: {train_mse:.3f}\n"
            f"Test MSE: {test_mse:.3f}\n"
            f"Training MAE: {train_mae:.3f}\n"
            f"Test MAE: {test_mae:.3f}\n"
            f"Training R-squared: {train_r2:.3f}\n"
            f"Test R-squared: {test_r2:.3f}\n"
        )

        # Save the summary to a temporary file and log it
        with tempfile.NamedTemporaryFile(mode='w+', suffix="__regression report.txt", delete=False) as tmp:
            tmp.write(summary_text)
            tmp.flush()

            # Log the temporary file as an artifact
            mlflow.log_artifact(tmp.name, 'Metrics')

        # Remove the temporary file
        os.remove(tmp.name)

    def log_qq_plot(self, pipeline, X_test, y_test):
        """
        Generates and logs a Q-Q plot as an MLflow artifact to assess normality of residuals.

        Parameters:
        pipeline (sklearn.pipeline.Pipeline): Object type that implements the "fit" and "predict" methods
        X_test, y_test (pd.DataFrame): Test dataset (features and target).
        """
        residuals = y_test - pipeline.predict(X_test)
        plt.figure(figsize=(8, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot')

        with tempfile.NamedTemporaryFile(suffix="__qq.png", delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()

            mlflow.log_artifact(tmp.name, 'Metrics')
            os.remove(tmp.name)

    def log_scale_location_plot(self, pipeline, X_test, y_test):
        """
        Generates and logs a scale-location plot as an MLflow artifact to check homoscedasticity.

        Parameters:
        pipeline (sklearn.pipeline.Pipeline): Object type that implements the "fit" and "predict" methods
        X_test, y_test (pd.DataFrame): Test dataset (features and target).
        """
        y_pred = pipeline.predict(X_test)
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.5)
        plt.xlabel('Predicted values')
        plt.ylabel('Sqrt(Absolute Residuals)')
        plt.title('Scale-Location Plot')

        with tempfile.NamedTemporaryFile(suffix="__scale_location.png", delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()

            mlflow.log_artifact(tmp.name, 'Metrics')
            os.remove(tmp.name)
