import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
import logging
import torch

from src.ModelManager import ModelManager

logger = logging.getLogger()
logging.getLogger("neo4j").setLevel(logging.INFO)

class EvaluationManager:
    def __init__(
        self,
        model: ModelManager,
        X_test: np.ndarray,
        y_test: np.ndarray,
        output_dir: str = "evaluation_results",
    ):
        """
        Initialize with a ModelManager instance, test data arrays, and output directory.
        """
        if not isinstance(model, ModelManager):
            raise ValueError("EvaluationManager requires a ModelManager instance as model")
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def confusion_matrix(self):
        """
        Generate and save the confusion matrix plot.
        Special-case when there's only one true label.
        """
        unique_labels = np.unique(self.y_test)
        if unique_labels.size == 1:
            cm = np.array([[len(self.y_test)]])
            return cm

        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(unique_labels))
        plt.xticks(tick_marks)
        plt.yticks(tick_marks)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        save_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Confusion matrix saved at: {save_path}")
        return cm

    def classification_report(self):
        """
        Generate and save the classification report to a text file.
        Also expose top-level precision/recall/f1-score keys for tests.
        """
        y_pred = self.model.predict(self.X_test)

        report_dict = classification_report(
            self.y_test, y_pred, output_dict=True, zero_division=0
        )
        # Add top-level metrics
        macro = report_dict.get("macro avg", {})
        report_dict["precision"] = macro.get("precision", 0.0)
        report_dict["recall"]    = macro.get("recall",    0.0)
        report_dict["f1-score"]  = macro.get("f1-score",  0.0)

        report_str = classification_report(self.y_test, y_pred, zero_division=0)
        report_path = os.path.join(self.output_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report_str)
        logger.info(f"Classification report saved at: {report_path}")

        return report_dict

    def roc_curve(self):
        """
        Generate and save the ROC curve for binary or multi-class classification.
        """
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(self.X_test)
        else:
            # Fallback: use model.model for raw torch.nn.Module
            with torch.no_grad():
                X_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(self.model.device)
                outputs = self.model.model(X_tensor)
                y_prob = torch.softmax(outputs, dim=1).cpu().numpy()

        if len(np.unique(self.y_test)) > 2:
            for i in range(y_prob.shape[1]):
                fpr, tpr, _ = roc_curve(self.y_test == i, y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
        else:
            fpr, tpr, _ = roc_curve(self.y_test, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")

        save_path = os.path.join(self.output_dir, "roc_curve.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"ROC curve saved at: {save_path}")

    def accuracy(self):
        """
        Calculate and return accuracy.
        """
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        logger.info(f"Accuracy: {acc * 100:.2f}%")
        return acc

    def loss_curve(self, history):
        """
        Save the loss curve plot if history is available.
        """
        if isinstance(history, dict) and 'loss' in history and 'val_loss' in history:
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Loss Curve')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            save_path = os.path.join(self.output_dir, "loss_curve.png")
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Loss curve saved at: {save_path}")
        else:
            logger.info("Loss history not available or incorrect format.")
