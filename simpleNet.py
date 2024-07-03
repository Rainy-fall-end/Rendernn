from torch import nn
import torch
class NeuralNetwork(nn.Module):
    def __init__(self,in_size,out_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_size,in_size*2),
            nn.ReLU(),
            nn.Linear(in_size*2,in_size*4),
            nn.ReLU(),
            nn.Linear(in_size*4,in_size*8),
            nn.ReLU(),
            nn.Linear(in_size*8,in_size*16),
            nn.ReLU(),
            nn.Linear(in_size*16,in_size*8),
            nn.ReLU(),
            nn.Linear(in_size*8,in_size*4),
            nn.ReLU(),
            nn.Linear(in_size*4,in_size*2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_size*2, out_size),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.linear_relu_stack(x)
        x = torch.clamp(x,0,255)
        return x