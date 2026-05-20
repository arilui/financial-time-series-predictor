import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.data import download_stock_data, create_sequences
from app.model import LSTMForecaster


# Hyperparameters
SEQUENCE_LENGTH = 60
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001


# Load and preprocess data
df = download_stock_data("AAPL")

X, y, scaler = create_sequences(df, sequence_length=SEQUENCE_LENGTH)

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Train/test split
train_size = int(len(X_tensor) * 0.8)

X_train = X_tensor[:train_size]
y_train = y_tensor[:train_size]

X_test = X_tensor[train_size:]
y_test = y_tensor[train_size:]

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = LSTMForecaster()

# Loss + optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        predictions = model(batch_X)

        loss = criterion(predictions, batch_y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.6f}")

# Save model
torch.save(model.state_dict(), "lstm_model.pth")

print("\nTraining complete.")
print("Model saved as lstm_model.pth")