import os
import numpy as np
import pytest
from src.EvaluationManager import EvaluationManager
from src.ModelManager import ModelManager

@pytest.fixture
def sample_data():
    """
    Fixture to generate sample data for evaluation tests.
    """
    rng = np.random.default_rng(42)
    X = rng.random((100, 2), dtype=np.float32)
    y = rng.integers(0, 2, 100)
    return {'X': X, 'y': y}

@pytest.fixture
def eval_manager(tmp_path, sample_data):
    """
    Fixture to train a simple model and return an EvaluationManager with a temp output dir.
    """
    # Train a simple tabular model
    model_mgr = ModelManager(input_size=2, output_size=2)
    model_mgr.train(sample_data['X'], sample_data['y'], epochs=1)

    # Create a fresh output directory
    out_dir = tmp_path / "eval_results"
    out_dir.mkdir()

    # Initialize evaluator with this output directory
    evaluator = EvaluationManager(
        model=model_mgr,
        X_test=sample_data['X'],
        y_test=sample_data['y'],
        output_dir=str(out_dir)
    )
    return evaluator, out_dir

def test_confusion_matrix_files(eval_manager):
    evaluator, out_dir = eval_manager
    cm = evaluator.confusion_matrix()

    # Check shape and sum
    assert cm.shape == (2, 2)
    assert cm.sum() == 100

    # Verify file exists
    cm_path = out_dir / "confusion_matrix.png"
    assert cm_path.exists(), "Confusion matrix image not saved"

def test_classification_report_files(eval_manager):
    evaluator, out_dir = eval_manager
    report = evaluator.classification_report()

    # Check top-level keys
    assert 'precision' in report
    assert 'recall' in report
    assert 'f1-score' in report

    # Verify text file exists
    rpt_path = out_dir / "classification_report.txt"
    assert rpt_path.exists(), "Classification report text not saved"

def test_roc_curve_file(eval_manager):
    evaluator, out_dir = eval_manager
    evaluator.roc_curve()

    roc_path = out_dir / "roc_curve.png"
    assert roc_path.exists(), "ROC curve image not saved"

def test_accuracy_and_range(eval_manager):
    evaluator, _ = eval_manager
    acc = evaluator.accuracy()
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

def test_loss_curve_file(tmp_path, sample_data):
    model_mgr = ModelManager(input_size=2, output_size=2)
    model_mgr.train(sample_data['X'], sample_data['y'], epochs=1)
    out_dir = tmp_path / "loss_results"
    out_dir.mkdir()
    evaluator = EvaluationManager(
        model=model_mgr,
        X_test=sample_data['X'],
        y_test=sample_data['y'],
        output_dir=str(out_dir)
    )

    history = {'loss': [1.0, 0.5], 'val_loss': [0.8, 0.3]}
    evaluator.loss_curve(history)

    loss_path = out_dir / "loss_curve.png"
    assert loss_path.exists(), "Loss curve image not saved"

def test_confusion_matrix_all_same_class_file(eval_manager):
    evaluator, out_dir = eval_manager
    # Overwrite y_test to a single class and recompute
    evaluator.y_test = np.zeros_like(evaluator.y_test)

    cm = evaluator.confusion_matrix()
    assert cm.shape == (1, 1)
    assert cm[0, 0] == len(evaluator.y_test)

    # No file saved when only one class
    cm_path = out_dir / "confusion_matrix.png"
    assert not cm_path.exists(), "Confusion matrix should not be saved for single-class case"

def test_accuracy_random_predictions(eval_manager):
    evaluator, _ = eval_manager
    rng = np.random.default_rng(0)
    random_y = rng.integers(0, 2, evaluator.y_test.shape)
    evaluator.y_test = random_y
    acc = evaluator.accuracy()
    assert acc < 1.0

def test_model_persistence_file(eval_manager, tmp_path):
    evaluator, out_dir = eval_manager
    model_path = tmp_path / "model_test.pth"
    evaluator.model.save_model(str(model_path))

    new_mgr = ModelManager(input_size=2, output_size=2)
    new_mgr.load_model(str(model_path))
    new_eval = EvaluationManager(
        model=new_mgr,
        X_test=evaluator.X_test,
        y_test=evaluator.y_test,
        output_dir=str(out_dir)
    )
    acc1 = new_eval.accuracy()
    acc2 = evaluator.accuracy()
    assert np.isclose(acc1, acc2, atol=1e-5)
