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
        self.grid = nn.Parameter(
            torch.empty(grid_size[0], grid_size[1], self.feature_num, device=device, requires_grad=True),
            requires_grad=True
        )
        init.xavier_uniform_(self.grid)
        self.outlayers = nn.Sequential(
                    nn.Linear(self.feature_num,self.feature_num*4),
                    nn.LeakyReLU(),
                    nn.Linear(self.feature_num*4,3)
                )
    def bilinear_interpolate(self, vec):
        vec = vec.clone()  # Avoid modifying the input directly
        vec[:, 0] = vec[:, 0] / math.pi * (self.grid_size[0] - 1)
        vec[:, 1] = (vec[:, 1] + math.pi) / (2 * math.pi) * (self.grid_size[1] - 1)
        
        W, H = self.grid_size[1] - 1, self.grid_size[0] - 1
        top_left_x = vec[:, 0].long().to(self.device)
        top_left_y = vec[:, 1].long().to(self.device)

        x_fract = vec[:, 0] % 1
        y_fract = vec[:, 1] % 1

        bottom_right_x = torch.clamp(top_left_x + 1, max=W)
        bottom_right_y = torch.clamp(top_left_y + 1, max=H)

        tl_vectors = self.grid[top_left_y, top_left_x]    # Top-left corner vectors
        tr_vectors = self.grid[top_left_y, bottom_right_x]  # Top-right corner vectors
        bl_vectors = self.grid[bottom_right_y, top_left_x]  # Bottom-left corner vectors
        br_vectors = self.grid[bottom_right_y, bottom_right_x]  # Bottom-right corner vectors

        top_interp = (1 - x_fract).unsqueeze(1) * tl_vectors + x_fract.unsqueeze(1) * tr_vectors
        bottom_interp = (1 - x_fract).unsqueeze(1) * bl_vectors + x_fract.unsqueeze(1) * br_vectors
        interpolated_vectors = (1 - y_fract).unsqueeze(1) * top_interp + y_fract.unsqueeze(1) * bottom_interp

        return interpolated_vectors

    def forward(self, pos):
        x = self.bilinear_interpolate(pos)
        x = torch.sigmoid(x)
        x = x*255
        return x

class GridNetDir(nn.Module):
    def __init__(self, grid_size, feature_num, device):
        super(GridNetDir, self).__init__()
        self.feature_num = feature_num
        self.device = device
        self.grid_size = grid_size
        self.grid_pos_range = [[0, math.pi], [-math.pi, math.pi]]
        self.grid_dir_range = [[0, math.pi], [-math.pi, math.pi]]
        self.grid_pos = nn.Parameter(
            torch.empty(grid_size[0], grid_size[1], self.feature_num, device=device, requires_grad=True),
            requires_grad=True
        )
        self.grid_dir = nn.Parameter(
            torch.empty(grid_size[0], grid_size[1], self.feature_num, device=device, requires_grad=True),
            requires_grad=True
        )
        init.xavier_uniform_(self.grid_dir)
        init.xavier_uniform_(self.grid_pos)
        self.outlayers = nn.Sequential(
                    nn.Linear(self.feature_num*2,self.feature_num*4),
                    nn.LeakyReLU(),
                    nn.Linear(self.feature_num*4,3)
                )
    def bilinear_interpolate(self, grid,range,vec):
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)
        vec = vec.clone()  # Avoid modifying the input directly
        vec[:, 0] = (vec[:, 0]-range[0][0])/(range[0][1]-range[0][0])  * (self.grid_size[0] - 1)
        vec[:, 1] = (vec[:, 1]-range[1][0])/(range[1][1]-range[1][0])  * (self.grid_size[1] - 1)
        
        W, H = self.grid_size[1] - 1, self.grid_size[0] - 1
        top_left_x = vec[:, 0].long().to(self.device)
        top_left_y = vec[:, 1].long().to(self.device)

        x_fract = vec[:, 0] % 1
        y_fract = vec[:, 1] % 1

        bottom_right_x = torch.clamp(top_left_x + 1, max=W)
        bottom_right_y = torch.clamp(top_left_y + 1, max=H)

        tl_vectors = grid[top_left_y, top_left_x]    # Top-left corner vectors
        tr_vectors = grid[top_left_y, bottom_right_x]  # Top-right corner vectors
        bl_vectors = grid[bottom_right_y, top_left_x]  # Bottom-left corner vectors
        br_vectors = grid[bottom_right_y, bottom_right_x]  # Bottom-right corner vectors

        top_interp = (1 - x_fract).unsqueeze(1) * tl_vectors + x_fract.unsqueeze(1) * tr_vectors
        bottom_interp = (1 - x_fract).unsqueeze(1) * bl_vectors + x_fract.unsqueeze(1) * br_vectors
        interpolated_vectors = (1 - y_fract).unsqueeze(1) * top_interp + y_fract.unsqueeze(1) * bottom_interp

        return interpolated_vectors

    def forward(self, pos, dir):
        x_pos = self.bilinear_interpolate(self.grid_pos,self.grid_pos_range,pos)
        x_dir = self.bilinear_interpolate(self.grid_dir,self.grid_dir_range,dir)
        x = torch.cat((x_pos, x_dir), dim=1)
        x = self.outlayers(x)
        x = torch.sigmoid(x)
        x = x*255
        return x

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = GridNet([100,100],3,device)
    # test = torch.tensor([[math.pi/2,math.pi],[math.pi/4,math.pi/2]]).to(device)
    net = GridNetDir([100,100],3,device).to(device)
    test_pos = torch.tensor([math.pi/4,math.pi/2]).to(device)
    test_dir = torch.tensor([0.6,0.7]).to(device)
    net(test_pos,test_dir)