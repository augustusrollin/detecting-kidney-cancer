import os
import pytest

from EvaluationManager import EvaluationManager


# @pytest.mark.skip(reason="takes a while to run")
def test_evaluate():
    data_folder = "kidney_ct_data"

    os.makedirs(data_folder, exist_ok=True)

    csv_path = 'kidneyData.csv'
    img_dir = 'kidney_ct_data'  

    evaluation_manager = EvaluationManager(csv_path, img_dir)
    evaluation_manager.evaluate_model(csv_path, img_dir)
