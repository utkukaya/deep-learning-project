import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


class LogisticRegressionForBC(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, 512)
        self.linear_2 = nn.Linear(512, 128)
        self.linear_3 = nn.Linear(128, 32)
        self.linear_4 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1)
        x = self.relu(self.linear_1(x))
        x = self.relu(self.linear_2(x))
        x = self.relu(self.linear_3(x))
        x = self.linear_4(x)
        return x
