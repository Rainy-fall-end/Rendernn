import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init


class DirCNN(nn.Module):
    def __init__(self,neighbor_size):
        super(DirCNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3*neighbor_size*neighbor_size)
        )
        self.neighbor_size = neighbor_size
    def forward(self, x):
        x = self.fc(x)
        x = F.softmax(x.view(x.shape[0], -1), dim=1).view(-1 ,3, self.neighbor_size, self.neighbor_size)
        return x


class GridNet(nn.Module):
    def __init__(self, grid_size,neighbor_size,feature_num, device):
        super(GridNet, self).__init__()
        self.dirCNN = DirCNN(neighbor_size=neighbor_size).to(device)
        self.feature_num = feature_num
        self.neighbor_size = neighbor_size
        self.device = device
        self.grid_size = grid_size
        self.grid_range = [[0, math.pi], [-math.pi, math.pi]]
        self.grid_pos = nn.Parameter(
            torch.empty(
                grid_size[0] + neighbor_size -1,
                grid_size[1] + neighbor_size -1,
                self.feature_num,
                device=device,
                requires_grad=True,
            ),
            requires_grad=True,
        )
        init.xavier_uniform_(self.grid_pos)
        
    def get_neighbor(self, vec):
        vec = vec.clone()  # Avoid modifying the input directly
        vec[:, 0] = vec[:, 0] / math.pi * (self.grid_size[0] - 1)
        vec[:, 1] = (vec[:, 1] + math.pi) / (2 * math.pi) * (self.grid_size[1] - 1)
        offset = (self.neighbor_size-1)//2
        top_left_x = vec[:, 0].long().to(self.device) + offset
        top_left_y = vec[:, 1].long().to(self.device) + offset
        size = self.neighbor_size
        half_size = size // 2
        
        neighborhood_list = []

        for i in range(-half_size, half_size + 1):
            row_list = []
            for j in range(-half_size, half_size + 1):
                row_list.append(self.grid_pos[top_left_x + i, top_left_y + j])
            neighborhood_list.append(torch.stack(row_list, dim=2))
        
        vector_res = torch.stack(neighborhood_list, dim=2)
        return vector_res

    def forward(self, pos):
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
        pos_weight = self.dirCNN(pos)
        pos_neighbor = self.get_neighbor(pos)
        x = pos_weight*pos_neighbor
        x = x.sum(dim=[2, 3]) 
        x = torch.sigmoid(x)
        x = F.threshold(x, 0.1, 0)
        x = x * 255
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = GridNet([10, 10], 3,3, device=device).to(device)
    test_pos = torch.tensor(
        [[0.8 * math.pi, 0.8 * math.pi], [0.6 * math.pi, 0.6 * math.pi]]
    ).to(device)
    test_dir = torch.tensor(
        [[0.8 * math.pi, 0.8 * math.pi], [0.6 * math.pi, 0.6 * math.pi]]
    ).to(device)
    net(test_pos)
