import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)