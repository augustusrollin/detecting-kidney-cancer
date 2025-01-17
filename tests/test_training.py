import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from ModelManager import ModelManager
from sklearn.metrics import accuracy_score

# Create a small mock dataset for testing purposes
@pytest.fixture
def mock_data():
    # Create a simple synthetic dataset with features (X) and labels (y)
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 4, 100)  # 4 classes (normal, tumor, stone, cyst)
    return X, y

# Test to check if the model can be created successfully
def test_model_creation():
    model_manager = ModelManager()
    model = model_manager.create_model()
    assert model is not None, "Model creation failed"

# Test if the training process works
def test_model_training(mock_data):
    X, y = mock_data
    model_manager = ModelManager()
    
    # Split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model_manager.train(X_train, y_train, epochs=5)
    
    # Evaluate the model
    model_predictions = model_manager.evaluate(X_test, y_test)
    
    # Test if the model makes predictions
    assert model_predictions is not None, "Model evaluation failed"
    
    # Check if the predictions match the test set size
    assert len(model_predictions) == len(y_test), f"Expected {len(y_test)} predictions, got {len(model_predictions)}"

# Test that the model training improves over epochs
def test_training_improvement(mock_data):
    X, y = mock_data
    model_manager = ModelManager()
    
    # Split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model for multiple epochs
    model_manager.train(X_train, y_train, epochs=10)
    
    # Get the model's performance on the test set
    initial_accuracy = model_manager.evaluate(X_test, y_test)
    
    # Check if accuracy is greater than 0 (meaning the model learned something)
    assert initial_accuracy > 0, "Model did not improve during training"

# Test if the model is performing reasonably (accuracy > random chance)
def test_model_performance(mock_data):
    X, y = mock_data
    model_manager = ModelManager()
    
    # Split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model_manager.train(X_train, y_train, epochs=5)
    
    # Evaluate the model
    accuracy = model_manager.evaluate(X_test, y_test)
    
    # Ensure that the model's accuracy is greater than random chance
    random_accuracy = 1 / len(np.unique(y))  # For 4 classes, random accuracy is 0.25
    assert accuracy > random_accuracy, f"Model accuracy {accuracy} is lower than random chance"

# Test edge case where input data has no features
def test_empty_data():
    model_manager = ModelManager()
    
    X_empty = np.array([])
    y_empty = np.array([])
    
    with pytest.raises(ValueError):
        model_manager.train(X_empty, y_empty, epochs=5)

# Test edge case where training data and labels do not match in size
def test_mismatched_data_labels():
    model_manager = ModelManager()
    
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 4, 80)  # Only 80 labels, should be 100
    
    with pytest.raises(ValueError):
        model_manager.train(X, y, epochs=5)
