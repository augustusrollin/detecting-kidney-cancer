import os
import pytest

from EvaluationManager import EvaluationManager


# @pytest.mark.skip(reason="takes a while to run")
def test_evaluate(data_folder, csv_path):
    os.makedirs(data_folder, exist_ok=True)
    evaluation_manager = EvaluationManager(csv_path, data_folder)
    evaluation_manager.evaluate_model(csv_path, data_folder)