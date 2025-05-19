import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ModelManager:
    def __init__(self, input_size, output_size, device=None):
        """
        Initializes the model manager for tabular or image data.
        """
        # Determine device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Distinguish between tabular and image input
        if isinstance(input_size, int) or (
            isinstance(input_size, (tuple, list)) and len(input_size) == 1
        ):
            # Tabular data (e.g. your 2-feature tests)
            self.is_tabular = True
            self.input_dim = (
                input_size if isinstance(input_size, int) else input_size[0]
            )
            self.model = nn.Linear(self.input_dim, output_size)
        else:
            # Image data (e.g. 3×224×224 for ResNet)
            self.is_tabular = False
            self.input_shape = input_size  # expected as tuple (C, H, W)
            self.model = models.resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V1
            )
            self.model.fc = nn.Linear(self.model.fc.in_features, output_size)

        # Move model to the selected device
        self.model = self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=1e-5
        )

    def _validate_input(self, X):
        """
        Validates and preprocesses input data based on model type.
        """
        X = torch.tensor(X, dtype=torch.float32)
        if self.is_tabular:
            # Tabular: expect (batch_size, input_dim)
            if X.dim() != 2 or X.shape[1] != self.input_dim:
                raise ValueError(
                    f"Expected tabular input shape (batch_size, {self.input_dim}), "
                    f"but got {X.shape}"
                )
        else:
            # Image: convert HWC->CHW if needed
            if X.dim() == 4 and X.shape[-1] == 3 and X.shape[1] != 3:
                X = X.permute(0, 3, 1, 2)
            # Then enforce (batch_size, 3, 224, 224)
            if X.dim() != 4 or X.shape[1:] != (3, 224, 224):
                raise ValueError(
                    f"Expected image input shape (batch_size, 3, 224, 224), "
                    f"but got {X.shape}"
                )
        return X

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Trains the model.
        """
        X_train = self._validate_input(X_train)
        y_train = torch.tensor(y_train, dtype=torch.long)

        # Ensure data size matches
        assert len(X_train) == len(y_train), "Mismatch between input and label sizes!"

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
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
    
    # In src/ModelManager.py, add:

    def predict_proba(self, X):
        self.model.eval()
        import torch
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        