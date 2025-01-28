import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class ModelManager:
    def __init__(self, input_size, output_size, learning_rate=0.001):
        """
        Initializes the model with input/output sizes and learning rate.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.history = {'loss': [], 'val_loss': []}  # Store loss history for plotting

    def _build_model(self):
        """
        Build a simple feedforward neural network.
        """
        return nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=250, batch_size=32):
        """
        Train the model using provided data with optional validation data.
        """
        # Ensure correct input size
        if len(X_train) != len(y_train):
            raise ValueError("Size mismatch between features and labels")

        # Convert data to tensors
        X_tensor = torch.Tensor(X_train)
        y_tensor = torch.LongTensor(y_train)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Validation set if provided
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.Tensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.history['loss'].append(avg_loss)

            # Validation loss calculation
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()
                    self.history['val_loss'].append(val_loss)
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
                self.model.train()
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model and return accuracy.
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.Tensor(X_test)
            y_tensor = torch.LongTensor(y_test)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, axis=1)
            accuracy = (predictions == y_tensor).float().mean().item()
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def predict(self, X):
        """
        Make predictions on new data.
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.Tensor(X)
            outputs = self.model(X_tensor)
            return torch.argmax(outputs, axis=1).numpy()

    def save_model(self, filepath):
        """
        Save model to a file.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'output_size': self.output_size,
            'history': self.history
        }, filepath)

    def load_model(self, filepath):
        """
        Load model from a file.
        """
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.input_size = checkpoint['input_size']
        self.output_size = checkpoint['output_size']
        self.history = checkpoint.get('history', {'loss': [], 'val_loss': []})
        self.model.eval()

    def get_loss_history(self):
        """
        Retrieve the stored training loss history.
        """
        return self.history
    