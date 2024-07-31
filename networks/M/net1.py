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
    def positionEncoder(self,L,x):
        y = torch.cat((torch.sin(2 * torch.pi * x), torch.cos(2 * torch.pi * x)), dim=0)
        for i in range(1,L):
            y = torch.cat((y,torch.sin(2**i * torch.pi * x), torch.cos(2**i * torch.pi * x)),dim=0)
        return y
    def forward(self, x):
        x = self.linear_relu_stack(x)
        x = torch.sigmoid(x)
        x = x*255
        return x
