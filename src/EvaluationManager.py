from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from DataPreprocessingManager import load_data, preprocess_data
import cv2

class EvaluationManager:

    def __init__(self, csv_path, img_dir):
        self.csv_path = csv_path
        self.img_dir = img_dir

    def evaluate_model(self, model_path='kidney_cnn_model.h5', training_history=None):
        """
        Evaluate the trained CNN model on the test set.

        Args:
            model_path (str): Path to the trained model.
            training_history (dict): History object containing training and validation loss.
        """
        # Load and preprocess data
        images, labels = load_data(self.csv_path, self.img_dir)
        _, test_generator = preprocess_data(images, labels)

        # Load the model
        model = load_model(model_path)

        # Evaluate the model
        loss, accuracy = model.evaluate(test_generator)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        # Predict the classes
        predictions = model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(test_generator.labels, axis=1)

        # Print classification report
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Cyst', 'Normal', 'Stone', 'Tumor']))

        # Print confusion matrix
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        # ROC Curve and AUC for the "Tumor" class
        tumor_class_index = 3  # Adjust this index based on your class order
        tumor_scores = predictions[:, tumor_class_index]  # Get scores for the "Tumor" class
        fpr, tpr, thresholds = roc_curve(y_true == tumor_class_index, tumor_scores)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for Tumor Class')
        plt.legend(loc="lower right")
        plt.show()

        # Plot Loss Curves if training history is provided
        if training_history is not None:
            self.plot_loss_curves(training_history)

    def plot_loss_curves(self, history):
        """
        Plot training and validation loss curves.

        Args:
            history (dict): History object containing training and validation loss.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()
