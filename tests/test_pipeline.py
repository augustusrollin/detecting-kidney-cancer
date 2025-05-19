import shutil
from pathlib import Path
import pytest

from src.PipelineManager import PipelineManager

# Use real dataset folder at project root
DATA_DIR = Path(__file__).parent.parent / "kidney_ct_data"

def test_pipeline_run_and_artifacts():
    """
    Run the pipeline on the real dataset and verify accuracy and artifacts.
    """
    output_dir = Path("evaluation_results_pipeline_test")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    pm = PipelineManager(
        data_dir=str(DATA_DIR),
        image_size=(224, 224),
        apply_augmentation=False,
    )
    _, acc = pm.run(
        batch_size=32,
        test_size=0.2,
        output_dir=str(output_dir)
    )
    # Accuracy should be between 0 and 1
    assert isinstance(acc, float) and 0.0 <= acc <= 1.0

    # Verify saved artifacts
    assert (output_dir / "confusion_matrix.png").exists(), "Confusion matrix image not saved"
    assert (output_dir / "classification_report.txt").exists(), "Classification report text not saved"
    assert (output_dir / "roc_curve.png").exists(), "ROC curve image not saved"

def test_cli_invocation_and_artifacts(capsys, monkeypatch):
    """
    Run the CLI on the real dataset and verify output and artifacts.
    """
    output_dir = Path("evaluation_results_pipeline_cli_test")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    import sys
    monkeypatch.setattr(sys, 'argv', [
        'prog',
        '--data-dir', str(DATA_DIR),
        '--batch-size', '2',
        '--test-size', '0.2',
        '--output-dir', str(output_dir),
        '--no-augment'
    ])

    PipelineManager.cli()
    captured = capsys.readouterr()
    assert 'Test accuracy:' in captured.out

    # Verify saved artifacts
    assert (output_dir / "confusion_matrix.png").exists(), "Confusion matrix image not saved by CLI"
    assert (output_dir / "classification_report.txt").exists(), "Classification report not saved by CLI"
    assert (output_dir / "roc_curve.png").exists(), "ROC curve not saved by CLI"
