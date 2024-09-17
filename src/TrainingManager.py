from data_preprocessing import load_data, preprocess_data
from model import build_model
import os

def train_model(csv_path, img_dir, model_save_path='kidney_cnn_model.h5'):
    # Load and preprocess data
    images, labels = load_data(csv_path, img_dir)
    train_generator, test_generator = preprocess_data(images, labels)
    
    # Build the model
    model = build_model()
    
    # Train the model
    model.fit(train_generator, epochs=10, validation_data=test_generator)
    
    # Save the model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    csv_path = 'kidneyData.csv'
    img_dir = './'  # Root directory of the project
    train_model(csv_path, img_dir)
