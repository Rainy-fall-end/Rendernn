import torch
from torch.nn import functional as F
import torch.nn as nn
class Residual_block(nn.Module):
    def __init__(self, size) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(size,size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(size),
            nn.Linear(size,size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(size)
        )
    def forward(self,x):
        residual = x
        out = self.layer(x)
        out = residual + out 
        out = F.relu(out)
        return out

class res_net(nn.Module):
    def __init__(self, input_size, output_size,res_num=20) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.res_num = res_num
        self.bn1 = nn.BatchNorm1d(input_size)
        self.input_layers = nn.Sequential(
            nn.Linear(self.input_size,self.input_size*2),
            nn.LeakyReLU(),
            nn.Linear(self.input_size*2,self.input_size*4),
            nn.LeakyReLU(),
            nn.Linear(self.input_size*4,self.input_size*8),
            nn.LeakyReLU(),
            nn.Linear(self.input_size*8,self.input_size*16),
            nn.LeakyReLU()
        )
        self.bn4 = nn.BatchNorm1d(input_size*16)
        self.res_net = nn.Sequential(*self.res_block(input_size*16))
        self.out_layers = nn.Sequential(
            nn.Linear(self.input_size*16,self.input_size*8),
            nn.LeakyReLU(),
            nn.Linear(self.input_size*8,output_size)
        )
    def res_block(self,size):
        blk = []
        for i in range(self.res_num):
            blk.append(Residual_block(size))
        return blk
    def forward(self,x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.bn1(x)
        x = self.input_layers(x)
        x = self.bn4(x)
        x = self.res_net(x)
        x = self.out_layers(x)
        return x
