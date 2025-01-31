import numpy as np
from sklearn.model_selection import train_test_split
from src.ModelManager import ModelManager

class TrainingManager:
    def __init__(self, input_size, output_size, batch_size=32, epochs=50):
        """
        Initialize the TrainingManager with necessary parameters.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_manager = ModelManager(input_size, output_size)

    def run_training(self, X, y):
        """
        Train and evaluate the model.
        """ 
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model_manager.train(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)
        
        # Evaluate the model
        accuracy = self.model_manager.evaluate(X_test, y_test)
        
        return self.model_manager, accuracy
        