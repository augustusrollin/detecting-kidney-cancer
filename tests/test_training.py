import os
import pytest

from TrainingManager import TrainingManager


# @pytest.mark.skip(reason="takes a while to run")
def test_train(data_folder, csv_path):
    os.makedirs(data_folder, exist_ok=True)
    training_manager = TrainingManager(csv_path, data_folder)
    training_manager.train_model(csv_path, data_folder)

