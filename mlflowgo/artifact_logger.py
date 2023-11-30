import mlflow
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    classification_report)
from sklearn.model_selection import learning_curve, validation_curve
import numpy as np
import pandas as pd
import os
import tempfile


class ArtifactLogger:
    """
    A class to create and log artifacts.

    Methods:
        log_artifact(self, local_path, artifact_path=None): Logs the artifact for the run.
    """

    def __init__(self):
        pass

    def log_learning_curves(self, pipeline, X, y, cv, scoring):
        """
        Generates and logs learning curve plot as an MLflow artifact.

        Parameters:
        pipeline (sklearn.pipeline.Pipeline): object type that implements the "fit" and "predict" methods
        X (pd.DataFrame): Feature dataset.
        y (pd.DataFrame): Target values.
        cv (int, optional): Number of cross-validation splits.
        scoring(str): A str (see model evaluation documentation) or a scorer callable object/function.
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
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()

            # Log the temporary file as an artifact
            mlflow.log_artifact(tmp.name, 'Learning Curves')

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
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()

            # Log the temporary file as an artifact
            mlflow.log_artifact(tmp.name, 'Validation Curve')

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
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()
            mlflow.log_artifact(tmp.name, 'ROC Curve')
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
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()
            mlflow.log_artifact(tmp.name, 'ROC Curve')
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
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()

            # Log the temporary file as an artifact
            mlflow.log_artifact(tmp.name, 'Precision Recall Curve')

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
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plt.savefig(tmp.name, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(tmp.name, 'Feature Importance')
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
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plt.savefig(tmp.name, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(tmp.name, 'Confusion Matrix')
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
        with tempfile.NamedTemporaryFile(mode='w', suffix=".txt", delete=False) as tmp:
            tmp.write(report)
            tmp.flush()

            # Log the temporary file as an artifact
            mlflow.log_artifact(tmp.name, 'Classification Report')

        # Remove the temporary file
        os.remove(tmp.name)
