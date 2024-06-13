import torch
import torch.nn as nn
import tqdm


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class TorchMLPClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        learning_rate=0.001,
        num_epochs=50,
        batch_size=32,
        tqdm=False,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = MLP(input_size, hidden_size, output_size)
        self.tqdm = tqdm

    def _validate_input(self, X, y):
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc"],
            multi_output=True,
            dtype=(np.float64, np.float32),
            reset=True,
        )
        self.classes_ = sklearn.utils.multiclass.unique_labels(y)
        return X, y

    def fit(self, X, y):
        X, y = self._validate_input(X, y)

        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Train the model
        epoch_range = range(self.num_epochs)
        if self.tqdm:
            epoch_range = tqdm.tqdm(epoch_range)
        for epoch in epoch_range:
            for i in range(0, len(X), self.batch_size):
                # Forward pass
                outputs = self.model(X_tensor[i : i + self.batch_size])

                # Compute loss
                loss = criterion(outputs, y_tensor[i : i + self.batch_size])

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, X):
        # Convert data to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Forward pass and get predictions
        outputs = self.model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)

        # Convert predictions to numpy array and return
        return predicted.numpy()

    def predict_proba(self, X):
        # Convert data to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Forward pass and get softmax probabilities
        outputs = self.model(X_tensor)
        softmax = nn.Softmax(dim=1)
        probabilities = softmax(outputs).detach().numpy()

        # Return probabilities
        return probabilities
