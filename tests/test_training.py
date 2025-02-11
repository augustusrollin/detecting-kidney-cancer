import pytest
import torch
import numpy as np
from src.ModelManager import ModelManager
from sklearn.model_selection import train_test_split

# Constants
INPUT_SIZE = (3, 224, 224)  # Match ResNet input size
OUTPUT_SIZE = 4  # Number of classes
BATCH_SIZE = 32
EPOCHS = 50

# Fix seed for reproducibility
rng = np.random.default_rng(seed=42)


@pytest.fixture
def mock_data():
    """
    Generate random tensor data similar to images for testing.
    """
    X = rng.random((100, *INPUT_SIZE), dtype=np.float32)
    y = rng.integers(0, OUTPUT_SIZE, size=100)
    return X, y


def test_model_creation():
    """
    Ensure the model initializes correctly.
    """
    model_manager = ModelManager(INPUT_SIZE, OUTPUT_SIZE)
    assert isinstance(model_manager, ModelManager)


def test_model_evaluation(mock_data):
    """
    Test the evaluation method of ModelManager.
    """
    X, y = mock_data
    model_manager = ModelManager(INPUT_SIZE, OUTPUT_SIZE)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_manager.train(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    accuracy = model_manager.evaluate(X_test, y_test)

    assert 0.0 <= accuracy <= 1.0, "Accuracy should be between 0 and 1"


def test_model_prediction(mock_data):
    """
    Test model prediction functionality.
    """
    X, _ = mock_data
    model_manager = ModelManager(INPUT_SIZE, OUTPUT_SIZE)

    # Train the model before making predictions
    model_manager.train(X, rng.integers(0, OUTPUT_SIZE, size=len(X)), epochs=EPOCHS, batch_size=BATCH_SIZE)

    predictions = model_manager.predict(X)
    assert len(predictions) == len(X), "Number of predictions should match input size"
    assert np.all(np.isin(predictions, np.arange(OUTPUT_SIZE))), "Predictions should be valid class indices"


def test_mismatched_data_labels():
    """
    Test the behavior of ModelManager when given mismatched input-output sizes.
    """
    model_manager = ModelManager(INPUT_SIZE, OUTPUT_SIZE)
    X = rng.random((100, *INPUT_SIZE), dtype=np.float32)
    y = rng.integers(0, OUTPUT_SIZE, size=80)  # Incorrect label count

    with pytest.raises(AssertionError, match="Mismatch between input and label sizes!"):
        model_manager.train(X, y, epochs=EPOCHS)


def test_model_save_load(mock_data, tmp_path):
    """
    Test model saving and loading functionality.
    """
    X, y = mock_data
    model_manager = ModelManager(INPUT_SIZE, OUTPUT_SIZE)

    # Train the model
    model_manager.train(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Save model to temporary path
    model_path = tmp_path / "test_model.pth"
    model_manager.save_model(str(model_path))

    # Load the saved model
    new_model_manager = ModelManager(INPUT_SIZE, OUTPUT_SIZE)
    new_model_manager.load_model(str(model_path))

    predictions = new_model_manager.predict(X)
    assert len(predictions) == len(X), "Loaded model should produce predictions of the same length"
