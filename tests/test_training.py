import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.ModelManager import ModelManager
from src.TrainingManager import TrainingManager

# Constants for input/output sizes
INPUT_SIZE = 10  # Assuming 10 features in synthetic data
OUTPUT_SIZE = 4  # Number of classes (e.g., Normal, Tumor, Stone, Cyst)

# Create a reproducible random number generator
rng = np.random.default_rng(seed=42)

@pytest.fixture
def mock_data():
    """
    Generates synthetic feature and label data for testing.
    """
    X = rng.random((100, INPUT_SIZE), dtype=np.float32)  # 100 samples, 10 features
    y = rng.integers(0, OUTPUT_SIZE, size=100)  # 100 labels in the correct range
    return X, y

def test_model_creation():
    """
    Test whether ModelManager is initialized correctly.
    """
    model_manager = ModelManager(INPUT_SIZE, OUTPUT_SIZE)
    assert hasattr(model_manager, "train"), "ModelManager should have a 'train' method"
    assert hasattr(model_manager, "evaluate"), "ModelManager should have an 'evaluate' method"

def test_model_training(mock_data):
    """
    Test the training process of ModelManager.
    """
    X, y = mock_data
    model_manager = ModelManager(INPUT_SIZE, OUTPUT_SIZE)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_manager.train(X_train, y_train, epochs=5, batch_size=16)
    assert model_manager is not None, "Training failed - model is None"

def test_model_evaluation(mock_data):
    """
    Test the evaluation method of ModelManager.
    """
    X, y = mock_data
    model_manager = ModelManager(INPUT_SIZE, OUTPUT_SIZE)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_manager.train(X_train, y_train, epochs=5, batch_size=16)
    accuracy = model_manager.evaluate(X_test, y_test)
    assert 0.0 <= accuracy <= 1.0, "Accuracy should be within [0,1]"

def test_model_prediction(mock_data):
    """
    Test model prediction functionality.
    """
    X, _ = mock_data
    model_manager = ModelManager(INPUT_SIZE, OUTPUT_SIZE)

    # Train the model before making predictions
    model_manager.train(X, rng.integers(0, OUTPUT_SIZE, size=len(X)), epochs=5, batch_size=16)
    
    predictions = model_manager.predict(X)
    assert predictions.shape[0] == X.shape[0], "Number of predictions should match input samples"
    assert np.all((predictions >= 0) & (predictions < OUTPUT_SIZE)), "Predictions should be within valid class range"

def test_mismatched_data_labels():
    """
    Test the behavior of ModelManager when given mismatched input-output sizes.
    """
    model_manager = ModelManager(INPUT_SIZE, OUTPUT_SIZE)
    X = rng.random((100, INPUT_SIZE), dtype=np.float32)
    y = rng.integers(0, OUTPUT_SIZE, size=80)  # Incorrect label count
    
    with pytest.raises(ValueError, match="Size mismatch between features and labels"):
        model_manager.train(X, y, epochs=5)

def test_model_save_load(mock_data, tmp_path):
    """
    Test model saving and loading functionality.
    """
    X, y = mock_data
    model_manager = ModelManager(INPUT_SIZE, OUTPUT_SIZE)
    
    # Train the model
    model_manager.train(X, y, epochs=5, batch_size=16)
    
    # Save model to temporary path
    model_path = tmp_path / "test_model.pth"
    model_manager.save_model(str(model_path))
    
    # Load the saved model
    new_model_manager = ModelManager(INPUT_SIZE, OUTPUT_SIZE)
    new_model_manager.load_model(str(model_path))
    
    predictions = new_model_manager.predict(X)
    assert predictions.shape[0] == X.shape[0], "Loaded model should produce correct number of predictions"
