import torch
import torch.nn as nn

class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultilayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 11)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        x = torch.flatten(x, 1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out



class MultilayerPerceptronForPCA(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultilayerPerceptronForPCA, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 11)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        x = torch.flatten(x, 1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out