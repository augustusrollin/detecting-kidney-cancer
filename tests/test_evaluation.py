import pytest
import numpy as np
from src.EvaluationManager import EvaluationManager
from src.ModelManager import ModelManager

@pytest.fixture
def sample_data():
    """
    Fixture to generate sample data for evaluation tests.
    """
    np.random.seed(42)
    X = np.random.rand(100, 2).astype(np.float32)  # 100 samples, 2 features
    y = np.random.randint(0, 2, 100)  # Binary classification labels (0 or 1)
    return {'X': X, 'y': y}

def test_confusion_matrix(sample_data):
    """
    Test confusion matrix generation and correctness.
    """
    model = ModelManager(input_size=2, output_size=2)
    model.train(sample_data['X'], sample_data['y'], epochs=1)
    evaluator = EvaluationManager(model.model, sample_data['X'], sample_data['y'])
    
    cm = evaluator.confusion_matrix()
    assert cm.shape == (2, 2), "Confusion matrix should have shape (2,2)"
    assert cm.sum() == 100, "Total samples in confusion matrix should match dataset size"

def test_classification_report(sample_data):
    """
    Test classification report generation.
    """
    model = ModelManager(input_size=2, output_size=2)
    model.train(sample_data['X'], sample_data['y'], epochs=1)
    evaluator = EvaluationManager(model.model, sample_data['X'], sample_data['y'])
    
    report = evaluator.classification_report()
    assert 'precision' in report, "Classification report should contain precision"
    assert 'recall' in report, "Classification report should contain recall"
    assert 'f1-score' in report, "Classification report should contain F1-score"

def test_roc_curve(sample_data):
    """
    Test ROC curve generation.
    """
    model = ModelManager(input_size=2, output_size=2)
    model.train(sample_data['X'], sample_data['y'], epochs=1)
    evaluator = EvaluationManager(model.model, sample_data['X'], sample_data['y'])
    
    evaluator.roc_curve()  # Should generate a plot without errors

def test_accuracy(sample_data):
    """
    Test accuracy calculation and expected range.
    """
    model = ModelManager(input_size=2, output_size=2)
    model.train(sample_data['X'], sample_data['y'], epochs=1)
    evaluator = EvaluationManager(model.model, sample_data['X'], sample_data['y'])
    
    acc = evaluator.accuracy()
    assert 0.0 <= acc <= 1.0, "Accuracy should be between 0 and 1"

def test_loss_curve(sample_data):
    """
    Test loss curve plotting during training.
    """
    model = ModelManager(input_size=2, output_size=2)
    model.train(sample_data['X'], sample_data['y'], epochs=500)
    evaluator = EvaluationManager(model.model, sample_data['X'], sample_data['y'])
    
    evaluator.loss_curve(model)  # Should generate a plot without errors

def test_confusion_matrix_all_same_class(sample_data):
    """
    Test confusion matrix when the model predicts only one class.
    """
    model = ModelManager(input_size=2, output_size=2)
    
    # Force model to predict only class '0'
    sample_data['y'][:] = 0  
    model.train(sample_data['X'], sample_data['y'], epochs=1)
    
    evaluator = EvaluationManager(model.model, sample_data['X'], sample_data['y'])
    cm = evaluator.confusion_matrix()
    
    assert cm[0, 0] == len(sample_data['y']), "All predictions should fall into one class"

def test_accuracy_random_predictions(sample_data):
    """
    Test accuracy with random predictions to check evaluation limits.
    """
    model = ModelManager(input_size=2, output_size=2)
    np.random.seed(42)
    random_y = np.random.randint(0, 2, len(sample_data['y']))  # Random binary predictions
    evaluator = EvaluationManager(model.model, sample_data['X'], random_y)
    
    acc = evaluator.accuracy()
    assert acc < 1.0, "Accuracy should not be 100% with random predictions"

def test_model_persistence(sample_data, tmp_path):
    """
    Test model save/load functionality and accuracy consistency.
    """
    model = ModelManager(input_size=2, output_size=2)
    model.train(sample_data['X'], sample_data['y'], epochs=1)
    
    model_path = tmp_path / "test_model.pth"
    model.save_model(str(model_path))
    
    new_model = ModelManager(input_size=2, output_size=2)
    new_model.load_model(str(model_path))
    
    evaluator = EvaluationManager(new_model.model, sample_data['X'], sample_data['y'])
    acc1 = evaluator.accuracy()
    
    evaluator_new = EvaluationManager(model.model, sample_data['X'], sample_data['y'])
    acc2 = evaluator_new.accuracy()
    
    assert np.isclose(acc1, acc2, atol=1e-5), "Accuracy should remain the same after loading the model"
