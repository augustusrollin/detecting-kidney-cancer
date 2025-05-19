import argparse
from typing import Tuple

import numpy as np

from DataPreprocessingManager import DataPreprocessingManager
from TrainingManager import TrainingManager


class PipelineManager:
    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        apply_augmentation: bool = False,
    ):
        """
        Initializes the pipeline with data directory and preprocessing settings.

        Args:
            data_dir: Path to dataset directory with class subfolders.
            image_size: Target size for image resizing (H, W).
            apply_augmentation: Whether to apply data augmentation.
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.apply_augmentation = apply_augmentation
        self.dpm: DataPreprocessingManager
        self.trainer: TrainingManager

    def load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads images and labels, then preprocesses into arrays.

        Returns:
            X: Feature array
            y: Label array
        """
        self.dpm = DataPreprocessingManager(
            data_directory=self.data_dir,
            image_size=self.image_size,
            apply_augmentation=self.apply_augmentation,
        )
        self.dpm.load_data()
        return self.dpm.preprocess()

    def run(
        self,
        batch_size: int = 32,
        epochs: int = 50,
    ) -> Tuple[TrainingManager, float]:
        """
        Runs full training and returns trained manager & test accuracy.

        Args:
            batch_size: Training batch size.
            epochs: Number of training epochs.

        Returns:
            model_manager: The trained ModelManager instance.
            accuracy: Test-set accuracy (0.0–1.0).
        """
        X, y = self.load_and_preprocess()
        self.trainer = TrainingManager(
            input_size=X.shape[1:],
            output_size=len(set(y)),
            batch_size=batch_size,
            epochs=epochs,
        )
        return self.trainer.run_training(X, y)

    @classmethod
    def cli(cls) -> None:
        """
        Command-line interface for running the pipeline.
        Parses args, runs training, and prints accuracy.
        """
        parser = argparse.ArgumentParser(
            description="Run the full ML pipeline: load data, train, and report accuracy."
        )
        parser.add_argument(
            "--data-dir",
            required=True,
            help="Path to dataset directory with class subfolders.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Training batch size.",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=50,
            help="Number of training epochs.",
        )
        parser.add_argument(
            "--no-augment",
            action="store_true",
            help="Disable data augmentation.",
        )
        args = parser.parse_args()

        pm = cls(
            data_dir=args.data_dir,
            image_size=(224, 224),
            apply_augmentation=not args.no_augment,
        )
        _, test_acc = pm.run(batch_size=args.batch_size, epochs=args.epochs)
        print(f"Test accuracy: {test_acc:.2%}")
