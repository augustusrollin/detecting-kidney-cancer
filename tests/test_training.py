import os
import pytest

from TrainingManager import TrainingManager


# @pytest.mark.skip(reason="takes a while to run")
def test_train():
    data_folder = "kidney_ct_data"

    os.makedirs(data_folder, exist_ok=True)

    csv_path = 'kidneyData.csv'
    img_dir = 'kidney_ct_data'  

    training_manager = TrainingManager(csv_path, img_dir)
    training_manager.train_model(csv_path, img_dir)
