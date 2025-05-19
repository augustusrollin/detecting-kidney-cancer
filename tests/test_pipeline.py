import os
import cv2
import numpy as np
import pytest
from src.PipelineManager import PipelineManager
from src.ModelManager import ModelManager


@pytest.fixture
def dummy_dataset(tmp_path):
    """
    Create a minimal dummy dataset with two classes and valid image files.
    """
    categories = ["Class0", "Class1"]
    for cat in categories:
        cat_dir = tmp_path / cat
        cat_dir.mkdir()
        # Generate 5 blank RGB images per category
        for i in range(5):
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            path = cat_dir / f"img_{i}.jpg"
            cv2.imwrite(str(path), img)
    return str(tmp_path)


def test_pipeline_run(dummy_dataset):
    """
    Test that PipelineManager.run trains a model and returns accuracy in [0,1].
    """
    pm = PipelineManager(
        data_dir=dummy_dataset,
        image_size=(224, 224),
        apply_augmentation=False,
    )
    model_mgr, acc = pm.run(batch_size=2, epochs=1)
    # Ensure a ModelManager is returned
    assert isinstance(model_mgr, ModelManager)
    # Accuracy should be a float between 0 and 1
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_cli_invocation(dummy_dataset, capsys, monkeypatch):
    """
    Test that PipelineManager.cli runs end-to-end and prints test accuracy.
    """
    import sys
    monkeypatch.setattr(sys, 'argv', [
        'prog',
        '--data-dir', dummy_dataset,
        '--batch-size', '2',
        '--epochs', '1',
        '--no-augment'
    ])
    PipelineManager.cli()
    captured = capsys.readouterr()
    assert 'Test accuracy:' in captured.out
