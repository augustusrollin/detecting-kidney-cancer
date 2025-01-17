import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ModelManager:
    def __init__(self, input_size, output_size):
        """
        Initialize the model, loss function, and optimizer.
        """
        self.model = self.build_model(input_size, output_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def build_model(self, input_size, output_size):
        """
        Build the neural network model.
        """
        model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_size)
        )
        return model

    def train(self, X_train, y_train, batch_size=32, epochs=10):
        """
        Train the model using the training data.
        """
        dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train).long())
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = torch.Tensor(X_test)
            labels = torch.Tensor(y_test).long()
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).float().mean()
            print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def save_model(self, path):
        """
        Save the trained model to a file.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """
        Load a pre-trained model from a file.
        """
        self.model.load_state_dict(torch.load(path))
