import torch
import torch.nn as nn


class Norm(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
        self.register_buffer('std', torch.ones(1, 3, 1, 1))
        self.mean.data = torch.FloatTensor(mean).view(self.mean.shape)
        self.std.data = torch.FloatTensor(std).view(self.std.shape)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x
