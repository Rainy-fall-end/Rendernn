from torch import nn
import torch
class NeuralNetwork(nn.Module):
    def __init__(self,in_size,out_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_size,in_size*128),
            nn.ReLU(),
            nn.Linear(in_size*128, out_size)
        )
    def forward(self, x):
        x = self.linear_relu_stack(x)
        x = torch.sigmoid(x)
        x = x*255
        return x
