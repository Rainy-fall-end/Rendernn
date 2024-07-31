import torch
import torch.nn as nn
import torch.nn.functional as F

class BnHitNet(nn.Module):
    def __init__(self,L,device):
        super(BnHitNet, self).__init__()
        self.L = L
        self.device = device
        # Common layers
        self.fc1 = nn.Linear(2*2*L, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 32)
        self.bn5 = nn.BatchNorm1d(32)
        # Classification head
        self.clsf1 = nn.Linear(32, 256)  # Binary classification for hit/miss
        self.clsb1 = nn.BatchNorm1d(256)
        self.clsf2 = nn.Linear(256, 128)  # Binary classification for hit/miss
        self.clsb2 = nn.BatchNorm1d(128)
        self.clsf3 = nn.Linear(128, 32)  # Binary classification for hit/miss
        self.clsb3 = nn.BatchNorm1d(32)
        self.clsf4 = nn.Linear(32, 1)
        # RGB prediction head
        self.rgbf1 = nn.Linear(32, 256)
        self.rgbb1 = nn.BatchNorm1d(256)
        self.rgbf2 = nn.Linear(256, 128)
        self.rgbb2 = nn.BatchNorm1d(128)
        self.rgbf3 = nn.Linear(128, 64)
        self.rgbb3 = nn.BatchNorm1d(64)
        self.rgbf4 = nn.Linear(64, 32)
        self.rgbb4 = nn.BatchNorm1d(32)
        self.rgbf5 = nn.Linear(32, 3)
    
    def positionEncoder(self,x):
        y = torch.cat((torch.sin(1 * torch.pi * x), torch.cos(1 * torch.pi * x)), dim=1)
        for i in range(1,self.L):
            y = torch.cat((y,torch.sin(2**i * torch.pi * x), torch.cos(2**i * torch.pi * x)),dim=1)
        return y
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.positionEncoder(x)
        # Common forward pass
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        # Classification head
        x_hit = F.relu(self.clsb1(self.clsf1(x)))
        x_hit = F.relu(self.clsb2(self.clsf2(x_hit)))
        x_hit = F.relu(self.clsb3(self.clsf3(x_hit)))
        x_hit = torch.sigmoid(self.clsf4(x_hit))
        # RGB prediction head
        x_rgb = F.relu(self.rgbb1(self.rgbf1(x)))
        x_rgb = F.relu(self.rgbb2(self.rgbf2(x_rgb)))
        x_rgb = F.relu(self.rgbb3(self.rgbf3(x_rgb)))
        x_rgb = F.relu(self.rgbb4(self.rgbf4(x_rgb)))
        x_rgb = torch.sigmoid(self.rgbf5(x_rgb))
        # Conditional RGB output based on hit/miss
        # rgb_output = x_rgb * x_hit + (1 - x_hit) * torch.tensor([0.0, 0.0, 1.0], device=self.device)
        hit_mask = x_hit > 0.5
        # rgb_output = x_rgb * hit_mask.float()
        rgb_output = x_rgb*255.0
        return x_hit, rgb_output
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = BnHitNet(3,device).to(device)
    test = torch.tensor([[0.6,0.7],[0.7,0.6]]).to(device)
    res = net(test)
    print(res.shape)