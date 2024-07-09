import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LSTM']

class LSTM(nn.Module):
    def __init__(self, input_size=256, input_channel=1, num_label=6, hidden_size=128, num_layers=1):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_length = input_size
        # LSTM's input_size should be the number of features per timestep
        # Assuming each timestep contains 'input_channel' features
        self.lstm = nn.LSTM(input_size=input_channel, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_label)

    def forward(self, x):
        """
        x: (batch, feature, sequence)
        """
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x = x.permute(0, 2, 1) # (batch, sequence, feature)
        x, (hn, cn) = self.lstm(x, (h0, c0))  # LSTM returns both hidden state and cell state
        x = x[:, -1, :] # (batch, feature)
        
        out = self.fc(x) # (batch, num_label)
        return out

if __name__ == '__main__':
    model = LSTM(input_size=256, input_channel=1)
    x = torch.randn(2, 1, 256)  # batch size, number of features, sequence length
    out = model(x)
    print(x.shape, out.shape)
