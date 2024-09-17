from tensorflow.keras.models import load_model
from src.DataPreprocessingManager import load_data, preprocess_data
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(csv_path, img_dir, model_path='kidney_cnn_model.h5'):
    """
    Evaluate the trained CNN model on the test set.

    Args:
        csv_path (str): Path to the CSV file containing image paths and labels.
        img_dir (str): Root directory where images are stored.
        model_path (str): Path to the trained model.
    """
    # Load and preprocess data
    images, labels = load_data(csv_path, img_dir)
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

if __name__ == '__main__':
    csv_path = 'kidneyData.csv'
    img_dir = 'kidney_ct_data'  # Adjusted to the correct directory
    evaluate_model(csv_path, img_dir)
