import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score

class EvaluationManager:
    def __init__(self, model, X_test, y_test, output_dir="evaluation_results"):
        """
        Initialize the evaluation manager with model, test data, and output directory.
        """
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
        """
        y_pred = self._get_predictions()
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(self.y_test)))
        plt.xticks(tick_marks)
        plt.yticks(tick_marks)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        save_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Confusion matrix saved at: {save_path}")
        return cm

    def classification_report(self):
        """
        Generate and save the classification report to a text file.
        """
        y_pred = self._get_predictions()
        report = classification_report(self.y_test, y_pred, output_dict=True)

        report_str = classification_report(self.y_test, y_pred)
        report_path = os.path.join(self.output_dir, "classification_report.txt")
        
        with open(report_path, "w") as f:
            f.write(report_str)

        print(f"Classification report saved at: {report_path}")
        return report

    def roc_curve(self):
        """
        Generate and save the ROC curve for binary classification.
        """
        y_prob = self._get_probabilities()
        
        fpr, tpr, _ = roc_curve(self.y_test, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")

        save_path = os.path.join(self.output_dir, "roc_curve.png")
        plt.savefig(save_path)
        plt.close()
        print(f"ROC curve saved at: {save_path}")

    def accuracy(self):
        """
        Calculate and print accuracy.
        """
        y_pred = self._get_predictions()
        acc = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {acc * 100:.2f}%")
        return acc

    def loss_curve(self, history):
        """
        Save the loss curve plot if history is available.
        """
        if isinstance(history, dict) and 'loss' in history and 'val_loss' in history:
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            save_path = os.path.join(self.output_dir, "loss_curve.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Loss curve saved at: {save_path}")
        else:
            print("Loss history not available or incorrect format.")

    def _get_predictions(self):
        """
        Internal method to get predictions safely.
        """
        if hasattr(self.model, 'predict'):
            return self.model.predict(self.X_test)
        elif hasattr(self.model, 'forward'):
            import torch
            with torch.no_grad():
                X_tensor = torch.Tensor(self.X_test)
                outputs = self.model.forward(X_tensor)
                return np.argmax(outputs.numpy(), axis=1)
        else:
            raise ValueError("Model must have a predict() or forward() method.")

    def _get_probabilities(self):
        """
        Internal method to get class probabilities safely.
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(self.X_test)
        elif hasattr(self.model, 'forward'):
            import torch
            with torch.no_grad():
                X_tensor = torch.Tensor(self.X_test)
                outputs = self.model.forward(X_tensor)
                return torch.softmax(outputs, dim=1).numpy()
        else:
            raise ValueError("Model must have a predict_proba() or forward() method.")