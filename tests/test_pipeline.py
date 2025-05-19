import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.PipelineManager import PipelineManager


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
            cv2.imwrite(str(cat_dir / f"img_{i}.jpg"), img)
    return str(tmp_path)


def test_pipeline_run_and_artifacts(dummy_dataset):
    """
    Test that PipelineManager.run trains a model, returns accuracy, and saves artifacts.
    """
    output_dir = Path("evaluation_results_pipeline_test")
    # Clean up previous runs
    if output_dir.exists():
        shutil.rmtree(output_dir)

    pm = PipelineManager(
        data_dir=dummy_dataset,
        image_size=(224, 224),
        apply_augmentation=False,
    )
    _, acc = pm.run(
        batch_size=2,
        epochs=1,
        test_size=0.2,
        output_dir=str(output_dir)
    )
    # Check accuracy range
    assert isinstance(acc, float) and 0.0 <= acc <= 1.0

    # Verify that evaluation artifacts were saved
    assert (output_dir / "confusion_matrix.png").exists(), "Confusion matrix image not saved"
    assert (output_dir / "classification_report.txt").exists(), "Classification report text not saved"
    assert (output_dir / "roc_curve.png").exists(), "ROC curve image not saved"


def test_cli_invocation_and_artifacts(dummy_dataset, capsys, monkeypatch):
    """
    Test that PipelineManager.cli runs end-to-end, prints accuracy, and saves artifacts.
    """
    output_dir = Path("evaluation_results_pipeline_cli_test")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    import sys
    monkeypatch.setattr(sys, 'argv', [
        'prog',
        '--data-dir', dummy_dataset,
        '--batch-size', '2',
        '--epochs', '1',
        '--test-size', '0.2',
        '--output-dir', str(output_dir),
        '--no-augment'
    ])

    PipelineManager.cli()
    captured = capsys.readouterr()
    assert 'Test accuracy:' in captured.out

    # Verify artifacts for CLI invocation
    assert (output_dir / "confusion_matrix.png").exists(), "Confusion matrix image not saved by CLI"
    assert (output_dir / "classification_report.txt").exists(), "Classification report not saved by CLI"
    assert (output_dir / "roc_curve.png").exists(), "ROC curve not saved by CLI"
