import mlflow
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score)
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

    def log_roc_curve(self, y_true, y_scores, feature_names):
        """ Determine which function to call to produce a ROC curve
        """
        if len(np.unique(y_true)) > 2:
            self._log_multi_class_roc_curve(y_true, y_scores, feature_names)
        else:
            self._log_binary_roc_curve(y_true, y_scores)

    def _plot_binary_roc_curve(self, y_true, y_scores):
        """ Plots a ROC curve for a binary classifier
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
        """ Plots a ROC curve for a multi-class classifier
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
        y_true: array-like of shape (n_samples,)
                True labels of the data.

        y_scores: array-like of shape (n_samples,) or (n_samples, n_classes)
                Target scores. Can either be probability estimates, confidence values, 
                or binary decisions.

        class_names: list
                    List of class names corresponding to the labels.
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
            mlflow.log_artifact(tmp.name, 'precision_recall_curve.png')

        # Remove the temporary file
        os.remove(tmp.name)

    def log_feature_importance(self, pipeline, model_step, feature_names):
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
        sample = pd.DataFrame(data).sample(n=sample_size)
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as tmp:
            sample.to_csv(tmp.name, index=False)
            mlflow.log_artifact(tmp.name, 'Data Sample')
        os.remove(tmp.name)
