import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init


class DirCNN(nn.Module):
    def __init__(self,neighbor_size,feature_num):
        super(DirCNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, feature_num*neighbor_size*neighbor_size)
        )
        self.neighbor_size = neighbor_size
        self.feature_num = feature_num
    def forward(self, x):
        x = self.fc(x)
        x = F.softmax(x.view(x.shape[0], -1), dim=1).view(-1 ,self.feature_num, self.neighbor_size, self.neighbor_size)
        return x


class GridNet(nn.Module):
    def __init__(self, grid_size,neighbor_size,feature_num, device):
        super(GridNet, self).__init__()
        self.dirnn = DirCNN(neighbor_size=neighbor_size,feature_num=feature_num).to(device)
        self.posnn = DirCNN(neighbor_size=neighbor_size,feature_num=feature_num).to(device)
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
        self.grid_dir = nn.Parameter(
            torch.empty(
                grid_size[0] + neighbor_size -1,
                grid_size[1] + neighbor_size -1,
                self.feature_num,
                device=device,
                requires_grad=True,
            ),
            requires_grad=True,
        )
        self.output_layers = nn.Sequential(
            nn.Linear(self.feature_num*2, 3)
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Linear(64, 3)
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

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        pos = x[:,:2]
        dir = x[:,2:]
        pos_neighbor = self.get_neighbor(pos)
        dir_neighbor = self.get_neighbor(dir)
        pos_weight = self.posnn(x)
        dir_weight = self.dirnn(x)
        x = torch.cat((pos_weight*pos_neighbor,dir_weight*dir_neighbor),dim=1)
        x = x.sum(dim=[2, 3])
        x = self.output_layers(x) 
        x = torch.sigmoid(x)
        # x = F.threshold(x, 0.1, 0.0)
        x = x * 255
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = GridNet([10, 10], 3,6, device=device).to(device)
    test_pos = torch.tensor(
        [[0.8 * math.pi, 0.8 * math.pi,0.8 * math.pi, 0.8 * math.pi], [0.6 * math.pi, 0.6 * math.pi,0.8 * math.pi, 0.8 * math.pi]]
    ).to(device)
    net(test_pos)
