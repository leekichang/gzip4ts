import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CNN']

class CNN(nn.Module):
    def __init__(self, input_size=1800, input_channel=1, num_label=6):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size = 1, stride = 1)
        self.pool1 = nn.AvgPool1d(kernel_size = 2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size = 1, stride = 1)
        self.pool2 = nn.AvgPool1d(kernel_size = 2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size = 1, stride = 1)
        self.pool3 = nn.AvgPool1d(kernel_size = 2)
        self.conv4 = nn.Conv1d(256, 128, kernel_size = 1, stride = 1)
        self.pool = nn.AvgPool1d(kernel_size = int(input_size / 8))
        self.fc = nn.Linear(128, num_label)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

if __name__ == '__main__':
    model = CNN(input_size=256)
    x = torch.randn(2, 1, 256)
    out = model(x)
    print(x.shape, out.shape)