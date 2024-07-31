import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
class GridNet(nn.Module):
    def __init__(self, grid_size, feature_num, device):
        super(GridNet, self).__init__()
        self.feature_num = feature_num
        self.device = device
        self.grid_size = grid_size
        self.grid_range = [[0, math.pi], [-math.pi, math.pi]]
        self.grid_pos = nn.Parameter(
            torch.empty(grid_size[0], grid_size[1], self.feature_num, device=device, requires_grad=True),
            requires_grad=True
        )
        self.grid_dir = nn.Parameter(
            torch.empty(50, 50, 5, 5, device=device, requires_grad=True),
            requires_grad=True
        )
        init.xavier_uniform_(self.grid_pos)
        # self.outlayers = nn.Sequential(
        #             nn.Linear(self.feature_num,self.feature_num*4),
        #             nn.LeakyReLU(),
        #             nn.Linear(self.feature_num*4,3)
        #         )
    def bilinear_interpolate(self, vec):
        vec = vec.clone()  # Avoid modifying the input directly
        vec[:, 0] = vec[:, 0] / math.pi * (self.grid_size[0] - 1)
        vec[:, 1] = (vec[:, 1] + math.pi) / (2 * math.pi) * (self.grid_size[1] - 1)
        
        W, H = self.grid_size[1] - 1, self.grid_size[0] - 1
        top_left_x = vec[:, 0].long().to(self.device)
        top_left_y = vec[:, 1].long().to(self.device)

        x_fract = vec[:, 0] % 1
        y_fract = vec[:, 1] % 1

        bottom_right_x = torch.where(top_left_x + 1 > W, torch.tensor(0, device=self.device), top_left_x + 1)
        bottom_right_y = torch.where(top_left_y + 1 > H, torch.tensor(0, device=self.device), top_left_y + 1)

        tl_vectors = self.grid_pos[top_left_y, top_left_x]    # Top-left corner vectors
        tr_vectors = self.grid_pos[top_left_y, bottom_right_x]  # Top-right corner vectors
        bl_vectors = self.grid_pos[bottom_right_y, top_left_x]  # Bottom-left corner vectors
        br_vectors = self.grid_pos[bottom_right_y, bottom_right_x]  # Bottom-right corner vectors

        top_interp = (1 - x_fract).unsqueeze(1) * tl_vectors + x_fract.unsqueeze(1) * tr_vectors
        bottom_interp = (1 - x_fract).unsqueeze(1) * bl_vectors + x_fract.unsqueeze(1) * br_vectors
        interpolated_vectors = (1 - y_fract).unsqueeze(1) * top_interp + y_fract.unsqueeze(1) * bottom_interp

        return interpolated_vectors

    def forward(self, pos, dir):
        x = self.bilinear_interpolate(pos)
        x = torch.sigmoid(x)
        x = F.threshold(x,0.1,0)
        x = x*255
        return x