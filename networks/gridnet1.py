import torch
import torch.nn as nn
import torch.nn.functional as F

def bilinear_interpolate(device,grid, normCoords):
    # Calculate the size of the grid
    W, H = grid.size(1) - 1, grid.size(0) - 1

    # Calculate the indices of the top-left corners
    top_left_x = (normCoords[:, 0] * W).long().to(device)
    top_left_y = (normCoords[:, 1] * H).long().to(device)

    # Calculate the fractional part of the indices
    x_fract = (normCoords[:, 0] * W) % 1
    y_fract = (normCoords[:, 1] * H) % 1

    # Calculate the indices of the corners
    bottom_right_x = torch.min(top_left_x + 1, torch.full_like(top_left_x, W, device=device))
    bottom_right_y = torch.min(top_left_y + 1, torch.full_like(top_left_y, H, device=device))


    # Gather the vectors at each corner for the entire batch
    tl_vectors = grid[top_left_y, top_left_x]    # Top-left corner vectors
    tr_vectors = grid[top_left_y, bottom_right_x]  # Top-right corner vectors
    bl_vectors = grid[bottom_right_y, top_left_x]  # Bottom-left corner vectors
    br_vectors = grid[bottom_right_y, bottom_right_x]  # Bottom-right corner vectors

    # Calculate the interpolated vectors
    top_interp = (1 - x_fract).unsqueeze(1) * tl_vectors + x_fract.unsqueeze(1) * tr_vectors
    bottom_interp = (1 - x_fract).unsqueeze(1) * bl_vectors + x_fract.unsqueeze(1) * br_vectors

    # Final interpolation between top and bottom
    interpolated_vectors = (1 - y_fract).unsqueeze(1) * top_interp + y_fract.unsqueeze(1) * bottom_interp

    return interpolated_vectors
class GridNet(nn.Module):
    def __init__(self,grid_size,feature_num,device):
        super(GridNet, self).__init__()
        self.feature_num = feature_num
        self.device = device
        self.grid_size = grid_size
        self.
        self.grid = nn.Parameter(torch.randn(grid_size[0], grid_size[1], self.feature_num, requires_grad=True,device=device)) * 10
        self.fc_layer = nn.Sequential(
            nn.Linear(self.feature_num*2,64),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(64,16),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(16,8),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(8,3),
            nn.Dropout(0.3),
            nn.LeakyReLU()
        )
    def bilinear_interpolate(self,vec):
        W,H = self.grid_size[1] - 1,self.grid_size[0] -1

    def forward(self, pos):
        x = bilinear_interpolate(self.device, self.pos_grid, pos).detach()  # Get features from position grid
        # Concatenate the features from both grids with the original input features
        x = self.fc_layer(x)
        x = torch.sigmoid(x)
        x = x*255
        return x
