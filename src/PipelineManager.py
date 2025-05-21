#!/usr/bin/env python3
"""
PipelineManager: orchestrates loading, preprocessing, training, evaluation, and artifact saving.
Encapsulates full ML pipeline and writes evaluation artifacts to disk.
Place this file in the `src/` directory alongside other managers.
"""
import argparse
from typing import Tuple

import numpy as np

from src.DataPreprocessingManager import DataPreprocessingManager
from src.ModelManager import ModelManager
from src.EvaluationManager import EvaluationManager


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
        self.dpm = DataPreprocessingManager(
            data_directory=self.data_dir,
            image_size=self.image_size,
            apply_augmentation=self.apply_augmentation,
        )

    def load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads images and labels, then preprocesses into arrays.

        Returns:
            X: Feature array
            y: Label array
        """
        self.dpm.load_data()
        return self.dpm.preprocess()

    def run(
        self,
        batch_size: int = 32,
        epochs: int = 3,
        test_size: float = 0.2,
        output_dir: str = "evaluation_results",
    ) -> Tuple[ModelManager, float]:
        """
        Executes the full pipeline: split data, train model, evaluate, and save artifacts.

        Args:
            batch_size: Training batch size.
            epochs: Number of training epochs.
            test_size: Fraction of data to reserve for test set.
            output_dir: Directory to save evaluation artifacts.

        Returns:
            model_mgr: The trained ModelManager instance.
            accuracy: Test-set accuracy (0.0–1.0).
        """
        # Load and preprocess
        X, y = self.load_and_preprocess()

        # Split into train/test
        X_train, X_test, y_train, y_test = self.dpm.split_data(test_size=test_size)

        # Determine ModelManager input_size: tabular vs image
        if X_train.ndim == 4 and X_train.shape[-1] == 3:
            # HWC format: (batch, H, W, C)
            _, H, W, C = X_train.shape
            input_size = (C, H, W)
        else:
            # Tabular or other
            input_size = X_train.shape[1] if X_train.ndim == 2 else X_train.shape[1:]

        # Train model
        model_mgr = ModelManager(input_size=input_size, output_size=len(set(y)))
        model_mgr.train(X_train, y_train, epochs=epochs, batch_size=batch_size)

        # Evaluate numeric accuracy
        accuracy = model_mgr.evaluate(X_test, y_test)

        # Save evaluation artifacts
        evaluator = EvaluationManager(
            model=model_mgr,
            X_test=X_test,
            y_test=y_test,
            output_dir=output_dir,
        )
        evaluator.confusion_matrix()
        evaluator.classification_report()
        evaluator.roc_curve()
        # Optionally save loss curve if history available
        history = getattr(model_mgr, 'history', None)
        if isinstance(history, dict):
            evaluator.loss_curve(history)

        return model_mgr, accuracy

    @classmethod
    def cli(cls) -> None:
        """
        Command-line interface for running the pipeline and saving artifacts.
        """
        parser = argparse.ArgumentParser(
            description="Run the full ML pipeline: load data, train, evaluate, and save artifacts."
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
            default=3,
            help="Number of training epochs.",
        )
        parser.add_argument(
            "--test-size",
            type=float,
            default=0.2,
            help="Fraction of data to reserve for testing.",
        )
        parser.add_argument(
            "--output-dir",
            default="evaluation_results",
            help="Directory to write evaluation artifacts.",
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
        _, test_acc = pm.run(
            batch_size=args.batch_size,
            epochs=args.epochs,
            test_size=args.test_size,
            output_dir=args.output_dir,
        )
        print(f"Test accuracy: {test_acc:.2%}")

    def run_on_arrays(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        batch_size: int = 32,
        epochs:     int = 3,
        output_dir: str = "evaluation_results"
    ) -> Tuple[ModelManager, float]:
        """
        Train & evaluate directly on provided arrays, saving artifacts.
        """
        # 1) Instantiate & train
        model_mgr = ModelManager(
            input_size=X_train.shape[1:], 
            output_size=len(np.unique(y_train))
        )
        history = model_mgr.train(X_train, y_train, epochs=epochs, batch_size=batch_size)

        # 2) Evaluate numeric accuracy
        acc = model_mgr.evaluate(X_test, y_test)
        print(f"Test accuracy: {acc:.2%}")

        # 3) Save artifacts
        evaluator = EvaluationManager(
            model=model_mgr,
            X_test=X_test,
            y_test=y_test,
            output_dir=output_dir,
        )
        evaluator.confusion_matrix()
        evaluator.classification_report()
        evaluator.roc_curve()
        evaluator.loss_curve(history)

        return model_mgr, acc
    