import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMForecaster, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        prediction = self.fc(last_output)
        return prediction


if __name__ == "__main__":
    model = LSTMForecaster()

    sample_input = torch.randn(32, 60, 1)
    sample_output = model(sample_input)

    print(model)
    print("Input shape:", sample_input.shape)
    print("Output shape:", sample_output.shape)