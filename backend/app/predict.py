import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.data import download_stock_data, create_sequences
from app.model import LSTMForecaster


SEQUENCE_LENGTH = 60
TRAIN_SPLIT = 0.8
TICKER = "AAPL"


df = download_stock_data(TICKER)
X, y, scaler = create_sequences(df, sequence_length=SEQUENCE_LENGTH)

X_tensor = torch.tensor(X, dtype=torch.float32)

train_size = int(len(X_tensor) * TRAIN_SPLIT)

X_test = X_tensor[train_size:]
y_test = y[train_size:]

model = LSTMForecaster()
model.load_state_dict(torch.load("lstm_model.pth"))
model.eval()

with torch.no_grad():
    test_predictions = model(X_test).numpy()

test_predictions = scaler.inverse_transform(test_predictions)
actual_test = scaler.inverse_transform(y_test)

mae = mean_absolute_error(actual_test, test_predictions)
rmse = np.sqrt(mean_squared_error(actual_test, test_predictions))

print(f"Ticker: {TICKER}")
print(f"Test MAE: ${mae:.2f}")
print(f"Test RMSE: ${rmse:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(actual_test, label="Actual Prices")
plt.plot(test_predictions, label="Predicted Prices")
plt.title(f"{TICKER} Test Set: Actual vs Predicted")
plt.xlabel("Test Time Step")
plt.ylabel("Price")
plt.legend()
plt.show()