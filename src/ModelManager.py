import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ModelManager:
    def __init__(self, input_size, output_size, device=None):
        """
        Initializes a model manager for tabular or image data.
        """
        # Determine device
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Distinguish between tabular and image input
        if isinstance(input_size, int) or (
            isinstance(input_size, (tuple, list)) and len(input_size) == 1
        ):
            # Tabular data path
            self.is_tabular = True
            self.input_dim = (
                input_size if isinstance(input_size, int) else input_size[0]
            )
            self.model = nn.Linear(self.input_dim, output_size)
        else:
            # Image data path
            self.is_tabular = False
            self.input_shape = input_size  # expected as (C, H, W)
            self.model = models.resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V1
            )
            self.model.fc = nn.Linear(self.model.fc.in_features, output_size)

        # Move model to selected device
        self.model = self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=1e-5
        )

    def _validate_input(self, X):
        """
        Convert and validate input array for tabular or image models.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if self.is_tabular:
            # Expect shape (batch_size, input_dim)
            if X_tensor.dim() != 2 or X_tensor.shape[1] != self.input_dim:
                raise ValueError(
                    f"Expected tabular input shape (batch_size, {self.input_dim}), but got {X_tensor.shape}"
                )
        else:
            # Handle HWC->CHW conversion if needed
            if X_tensor.dim() == 4 and X_tensor.shape[-1] == 3 and X_tensor.shape[1] != 3:
                X_tensor = X_tensor.permute(0, 3, 1, 2)
            # Enforce (batch_size, 3, H, W)
            if X_tensor.dim() != 4 or X_tensor.shape[1:] != self.input_shape:
                raise ValueError(
                    f"Expected image input shape (batch_size, {self.input_shape}), but got {X_tensor.shape}"
                )
        return X_tensor

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Train the model on provided data.
        """
        X_tensor = self._validate_input(X_train).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)

        # Ensure matching sizes
        assert len(X_tensor) == len(y_tensor), "Mismatch between input and label sizes!"

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )

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
        Evaluate the model on test data and return accuracy.
        """
        self.model.eval()
        X_tensor = self._validate_input(X_test).to(self.device)
        y_tensor = torch.tensor(y_test, dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predictions = torch.max(outputs, 1)

        accuracy = (predictions == y_tensor).float().mean().item()
        return accuracy

    def predict(self, X):
        """
        Return class predictions for input data.
        """
        self.model.eval()
        X_tensor = self._validate_input(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predictions = torch.max(outputs, 1)
        return predictions.cpu().numpy()

    def predict_proba(self, X):
        """
        Return class probabilities for input data.
        """
        self.model.eval()
        X_tensor = self._validate_input(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    def save_model(self, path):
        """Save model state to path."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Load model state from path."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.model.load_state_dict(torch.load(path, map_location=self.device))
    
