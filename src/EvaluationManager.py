from tensorflow.keras.models import load_model
from data_preprocessing import load_data, preprocess_data

def evaluate_model(csv_path, img_dir, model_path='kidney_cnn_model.h5'):
    # Load and preprocess data
    images, labels = load_data(csv_path, img_dir)
    _, test_generator = preprocess_data(images, labels)
    
    # Load the model
    model = load_model(model_path)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
if __name__ == '__main__':
    csv_path = 'kidneyData.csv'
    img_dir = './'  # Root directory of the project
    evaluate_model(csv_path, img_dir)
