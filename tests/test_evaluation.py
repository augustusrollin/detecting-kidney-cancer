import pytest
from src.EvaluationManager import EvaluationManager
from src.ModelManager import ModelManager

# @pytest.mark.skip(reason="takes a while to run")
def test_confusion_matrix(sample_data):
    """
    Test confusion matrix generation.
    """
    model = ModelManager(input_size=2, output_size=2)
    model.train(sample_data['X'], sample_data['y'], epochs=1)
    evaluator = EvaluationManager(model.model, sample_data['X'], sample_data['y'])
    evaluator.confusion_matrix()

# @pytest.mark.skip(reason="takes a while to run")
def test_classification_report(sample_data):
    """
    Test classification report generation.
    """
    model = ModelManager(input_size=2, output_size=2)
    model.train(sample_data['X'], sample_data['y'], epochs=1)
    evaluator = EvaluationManager(model.model, sample_data['X'], sample_data['y'])
    evaluator.classification_report()

# @pytest.mark.skip(reason="takes a while to run")
def test_roc_curve(sample_data):
    """
    Test ROC curve generation.
    """
    model = ModelManager(input_size=2, output_size=2)
    model.train(sample_data['X'], sample_data['y'], epochs=1)
    evaluator = EvaluationManager(model.model, sample_data['X'], sample_data['y'])
    evaluator.roc_curve()

# @pytest.mark.skip(reason="takes a while to run")
def test_accuracy(sample_data):
    """
    Test accuracy calculation.
    """
    model = ModelManager(input_size=2, output_size=2)
    model.train(sample_data['X'], sample_data['y'], epochs=1)
    evaluator = EvaluationManager(model.model, sample_data['X'], sample_data['y'])
    evaluator.accuracy()

# @pytest.mark.skip(reason="takes a while to run")
def test_loss_curve(sample_data):
    """
    Test loss curve plotting during training.
    """
    model = ModelManager(input_size=2, output_size=2)
    model.train(sample_data['X'], sample_data['y'], epochs=1)
    evaluator = EvaluationManager(model.model, sample_data['X'], sample_data['y'])
    evaluator.loss_curve(model)
