from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

from ModelManager import ModelManager
from DataPreprocessingManager import DataPreprocessingManager

class TrainingManager:

    def __init__(self, csv_path, img_dir, epochs):
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.epochs = epochs

    def train_model(self, model_save_path='kidney_cnn_model.h5', epochs=50):
        """
        Train the CNN model.

        Args:
            model_save_path (str): Path to save the trained model.
            epochs (int): Number of epochs for training.
        """
        # Create an instance of DataPreprocessingManager
        data_manager = DataPreprocessingManager(self.csv_path, self.img_dir, None, None)
        
        # Load and preprocess data
        images, labels = data_manager.load_data(self.csv_path, self.img_dir)
        train_generator, test_generator = data_manager.preprocess_data(images, labels)
        
        # Create an instance of ModelManager
        model_manager = ModelManager()
        
        # Build the model using the instance
        model = model_manager.build_model()
        
        # Define callbacks
        checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        
        # Train the model
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=test_generator,
            callbacks=[checkpoint, early_stopping]
        )
        
        # Save the final model
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")
    
    