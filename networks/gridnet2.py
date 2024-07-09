import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from torch.jit import script, trace
class GridNetDir(nn.Module):
    def __init__(self, grid_size, feature_num, device):
        super(GridNetDir, self).__init__()
        self.feature_num = feature_num
        self.device = device
        self.grid_size = grid_size
        self.grid_pos_range = torch.tensor([0, math.pi, -math.pi, math.pi], dtype=torch.float32)
        self.grid_dir_range = torch.tensor([0.5*math.pi, 0.85*math.pi, -0.85*math.pi, -0.5*math.pi], dtype=torch.float32)
        self.grid = nn.Parameter(
            torch.empty(grid_size[0], grid_size[1],grid_size[2], grid_size[3],self.feature_num, device=device, requires_grad=True),
            requires_grad=True
        )
        init.xavier_uniform_(self.grid)
        self.outlayers = nn.Sequential(
                    nn.Linear(self.feature_num,self.feature_num*4),
                    nn.LeakyReLU(),
                    nn.Linear(self.feature_num*4,3)
                )
    def bilinear_interpolate(self, vec):
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)

        # Normalize and map to grid indices
        vec[:, 0] = (vec[:, 0] - self.grid_pos_range[0]) / (self.grid_pos_range[1] - self.grid_pos_range[0]) * (self.grid_size[0] - 1)
        vec[:, 1] = (vec[:, 1] - self.grid_pos_range[2]) / (self.grid_pos_range[3] - self.grid_pos_range[2]) * (self.grid_size[1] - 1)
        vec[:, 2] = (vec[:, 2] - self.grid_dir_range[0]) / (self.grid_dir_range[1] - self.grid_dir_range[0]) * (self.grid_size[2] - 1)
        vec[:, 3] = (vec[:, 3] - self.grid_dir_range[2]) / (self.grid_dir_range[3] - self.grid_dir_range[2]) * (self.grid_size[3] - 1)

        W, H, D, T = self.grid_size[1] - 1, self.grid_size[0] - 1, self.grid_size[2] - 1, self.grid_size[3] - 1

        # Get indices of the top-left-front-near corner
        top_left_x = vec[:, 0].long().to(self.device)
        top_left_y = vec[:, 1].long().to(self.device)
        top_left_z = vec[:, 2].long().to(self.device)
        top_left_w = vec[:, 3].long().to(self.device)

        # Calculate fractional part
        x_fract = vec[:, 0] % 1
        y_fract = vec[:, 1] % 1
        z_fract = vec[:, 2] % 1
        w_fract = vec[:, 3] % 1

        # Get indices of the bottom-right-back-far corner
        bottom_right_x = torch.clamp(top_left_x + 1, max=W)
        bottom_right_y = torch.clamp(top_left_y + 1, max=H)
        bottom_right_z = torch.clamp(top_left_z + 1, max=D)
        bottom_right_w = torch.clamp(top_left_w + 1, max=T)

        # Gather the 8 corner vectors
        c0000 = self.grid[top_left_y, top_left_x, top_left_z, top_left_w]  # Top-left-front-near corner
        c1000 = self.grid[top_left_y, bottom_right_x, top_left_z, top_left_w]  # Top-right-front-near corner
        c0100 = self.grid[bottom_right_y, top_left_x, top_left_z, top_left_w]  # Bottom-left-front-near corner
        c0010 = self.grid[top_left_y, top_left_x, bottom_right_z, top_left_w]  # Top-left-back-near corner
        c0001 = self.grid[top_left_y, top_left_x, top_left_z, bottom_right_w]  # Top-left-front-far corner
        c1100 = self.grid[bottom_right_y, bottom_right_x, top_left_z, top_left_w]  # Bottom-right-front-near corner
        c1010 = self.grid[top_left_y, bottom_right_x, bottom_right_z, top_left_w]  # Top-right-back-near corner
        c1001 = self.grid[top_left_y, bottom_right_x, top_left_z, bottom_right_w]  # Top-right-front-far corner
        c0110 = self.grid[bottom_right_y, top_left_x, bottom_right_z, top_left_w]  # Bottom-left-back-near corner
        c0101 = self.grid[bottom_right_y, top_left_x, top_left_z, bottom_right_w]  # Bottom-left-front-far corner
        c0011 = self.grid[top_left_y, top_left_x, bottom_right_z, bottom_right_w]  # Top-left-back-far corner
        c1110 = self.grid[bottom_right_y, bottom_right_x, bottom_right_z, top_left_w]  # Bottom-right-back-near corner
        c1101 = self.grid[bottom_right_y, bottom_right_x, top_left_z, bottom_right_w]  # Bottom-right-front-far corner
        c1011 = self.grid[top_left_y, bottom_right_x, bottom_right_z, bottom_right_w]  # Top-right-back-far corner
        c0111 = self.grid[bottom_right_y, top_left_x, bottom_right_z, bottom_right_w]  # Bottom-left-back-far corner
        c1111 = self.grid[bottom_right_y, bottom_right_x, bottom_right_z, bottom_right_w]  # Bottom-right-back-far corner

        # Interpolate along x
        c00 = c0000 * (1 - x_fract).unsqueeze(1) + c1000 * x_fract.unsqueeze(1)
        c01 = c0001 * (1 - x_fract).unsqueeze(1) + c1001 * x_fract.unsqueeze(1)
        c10 = c0100 * (1 - x_fract).unsqueeze(1) + c1100 * x_fract.unsqueeze(1)
        c11 = c0110 * (1 - x_fract).unsqueeze(1) + c1110 * x_fract.unsqueeze(1)
        c20 = c0010 * (1 - x_fract).unsqueeze(1) + c1010 * x_fract.unsqueeze(1)
        c21 = c0011 * (1 - x_fract).unsqueeze(1) + c1011 * x_fract.unsqueeze(1)
        c30 = c0101 * (1 - x_fract).unsqueeze(1) + c1101 * x_fract.unsqueeze(1)
        c31 = c0111 * (1 - x_fract).unsqueeze(1) + c1111 * x_fract.unsqueeze(1)

        # Interpolate along y
        c0 = c00 * (1 - y_fract).unsqueeze(1) + c10 * y_fract.unsqueeze(1)
        c1 = c01 * (1 - y_fract).unsqueeze(1) + c11 * y_fract.unsqueeze(1)
        c2 = c20 * (1 - y_fract).unsqueeze(1) + c30 * y_fract.unsqueeze(1)
        c3 = c21 * (1 - y_fract).unsqueeze(1) + c31 * y_fract.unsqueeze(1)

        # Interpolate along z
        c4 = c0 * (1 - z_fract).unsqueeze(1) + c2 * z_fract.unsqueeze(1)
        c5 = c1 * (1 - z_fract).unsqueeze(1) + c3 * z_fract.unsqueeze(1)

        # Interpolate along w
        interpolated_vectors = c4 * (1 - w_fract).unsqueeze(1) + c5 * w_fract.unsqueeze(1)

        return interpolated_vectors

    def forward(self, x):
        # if pos.dim() == 1:
        #     pos = pos.unsqueeze(0)
        # if dir.dim() == 1:
        #     dir = dir.unsqueeze(0)    
        # x = torch.cat((pos, dir), dim=1)
        x = self.bilinear_interpolate(x)
        x = self.outlayers(x)
        x = torch.sigmoid(x)
        x = x*255
        return x
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = GridNetDir([64,64,64,64],3,device).to(device)
    test = torch.tensor([0.8*math.pi,0.8*math.pi,2,-2.53]).to(device)
    net(test)