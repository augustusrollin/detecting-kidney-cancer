import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ModelManager:
    def __init__(self, input_shape, output_size, device=None):
        """
        Initializes the model manager with a ResNet model.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # Fix deprecated warning

        # Modify the final layer to match the output size
        self.model.fc = nn.Linear(self.model.fc.in_features, output_size)
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Trains the model.
        """
        X_train = torch.tensor(X_train, dtype=torch.float32)

        # Ensure input has 4 dimensions (batch, channels, height, width)
        if len(X_train.shape) != 4 or X_train.shape[1:] != (3, 224, 224):
            raise ValueError(f"Expected input shape (batch, 3, 224, 224), but got {X_train.shape}")

        y_train = torch.tensor(y_train, dtype=torch.long)

        # Ensure data size matches
        assert len(X_train) == len(y_train), "Mismatch between input and label sizes!"

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for inputs, labels in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model.
        """
        self.model.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_test)
            _, predictions = torch.max(outputs, 1)

        accuracy = (predictions == y_test).float().mean().item()
        return accuracy

    def predict(self, X):
        """
        Makes predictions using the trained model.
        """
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X)
            _, predictions = torch.max(outputs, 1)

        return predictions.cpu().numpy()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
    
