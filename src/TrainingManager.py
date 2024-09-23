from srcs.ModelManager import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

from srcs.DataPreprocessingManager import load_data, preprocess_data

class TrainingManager:

    def __init__(self, csv_path, img_dir, epochs):
        csv_path = self.csv_path
        img_dir = self.img_dir
        epochs = self.epochs


    def train_model(self, csv_path, img_dir, model_save_path='kidney_cnn_model.h5', epochs=50):
        """
        Train the CNN model.

        Args:
            csv_path (str): Path to the CSV file containing image paths and labels.
            img_dir (str): Root directory where images are stored.
            model_save_path (str): Path to save the trained model.
            epochs (int): Number of epochs for training.
        """
        # Load and preprocess data
        images, labels = load_data(csv_path, img_dir)
        train_generator, test_generator = preprocess_data(images, labels)
        
        # Build the model
        model = build_model()
        
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
