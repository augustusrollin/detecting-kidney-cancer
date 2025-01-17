import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score

class EvaluationManager:
    def __init__(self, model, X_test, y_test):
        """
        Initialize the evaluation manager with model, test data, and test labels.
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def confusion_matrix(self):
        """
        Generate and plot the confusion matrix.
        """
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(self.y_test)))
        plt.xticks(tick_marks, np.unique(self.y_test), rotation=45)
        plt.yticks(tick_marks, np.unique(self.y_test))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def classification_report(self):
        """
        Generate and print the classification report including precision, recall, and F1-score.
        """
        y_pred = self.model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        print("Classification Report:")
        print(report)

    def roc_curve(self):
        """
        Generate and plot the ROC curve.
        """
        y_prob = self.model.predict_proba(self.X_test)
        fpr, tpr, _ = roc_curve(self.y_test, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()

    def accuracy(self):
        """
        Print the accuracy of the model on the test set.
        """
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {acc * 100:.2f}%")

    def loss_curve(self, history):
        """
        Plot the loss curve from the model training history.
        """
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='val')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
