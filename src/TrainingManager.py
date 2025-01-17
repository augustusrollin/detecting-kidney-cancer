import torch
from sklearn.model_selection import train_test_split
from src.ModelManager import ModelManager
from src.DataPreprocessingManager import DataPreprocessingManager

class TrainingManager:
    def __init__(self, data_directory, image_size=(224, 224)):
        """
        Initialize the Training Manager with dataset directory and image size.
        """
        self.data_directory = data_directory
        self.image_size = image_size

    def load_and_preprocess_data(self):
        """
        Load and preprocess data: resize, normalize, and split into train/test sets.
        Convert data into torch tensors for PyTorch compatibility.
        """
        data_manager = DataPreprocessingManager(self.data_directory, self.image_size)
        data_manager.load_data()  # Loads raw images and labels
        data, labels = data_manager.preprocess()  # Normalize and encode labels
        
        # Convert to torch tensors
        data = torch.Tensor(data)
        labels = torch.Tensor(labels).long()  # Use long type for classification
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self):
        """
        Train the model and return it alongside the accuracy on the test set.
        """
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()  # Get training and test data

        # Instantiate the model
        model_manager = ModelManager(X_train.shape[1], len(set(y_train)))  # Create the model based on input and output size
        model_manager.train(X_train, y_train, epochs=10)  # Train the model for 10 epochs

        accuracy = model_manager.evaluate(X_test, y_test)  # Evaluate the model on test data
        return model_manager, accuracy  # Return the model and its accuracy
