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
        # x = x.view(-1, 3, 3, 3)  # 假设 fc 的输出大小为 (batch_size, 256)
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
        # self.grid_dir = nn.Parameter(
        #     torch.empty(grid_size[1],grid_size[1], 3, 3, device=device, requires_grad=True),
        #     requires_grad=True
        # )
        init.xavier_uniform_(self.grid_pos)
        
    def get_neighbor(self, vec):
        vec = vec.clone()  # Avoid modifying the input directly
        vec[:, 0] = vec[:, 0] / math.pi * (self.grid_size[0] - 1)
        vec[:, 1] = (vec[:, 1] + math.pi) / (2 * math.pi) * (self.grid_size[1] - 1)
        offset = (self.neighbor_size-1)//2
        top_left_x = vec[:, 0].long().to(self.device) + offset
        top_left_y = vec[:, 1].long().to(self.device) + offset
        # if self.neighbor_size == 3:
        #     vector_res = torch.stack(
        #         (
        #             torch.stack(
        #                 (
        #                     self.grid_pos[top_left_x - 1, top_left_y + 1],
        #                     self.grid_pos[top_left_x, top_left_y + 1],
        #                     self.grid_pos[top_left_x + 1, top_left_y + 1],
        #                 ),
        #                 dim=2,
        #             ),
        #             torch.stack(
        #                 (
        #                     self.grid_pos[top_left_x - 1, top_left_y],
        #                     self.grid_pos[top_left_x, top_left_y],
        #                     self.grid_pos[top_left_x + 1, top_left_y],
        #                 ),
        #                 dim=2,
        #             ),
        #             torch.stack(
        #                 (
        #                     self.grid_pos[top_left_x - 1, top_left_y + 1],
        #                     self.grid_pos[top_left_x, top_left_y + 1],
        #                     self.grid_pos[top_left_x + 1, top_left_y + 1],
        #                 ),
        #                 dim=2,
        #             ),
        #         ),
        #         dim=2,
        #     )
        size = self.neighbor_size
        half_size = size // 2
        
        # 创建一个空列表来存储每个邻域的值
        neighborhood_list = []

        # 遍历每一行和每一列
        for i in range(-half_size, half_size + 1):
            row_list = []
            for j in range(-half_size, half_size + 1):
                row_list.append(self.grid_pos[top_left_x + i, top_left_y + j])
            # 堆叠当前行
            neighborhood_list.append(torch.stack(row_list, dim=2))
        
        # 堆叠所有行
        vector_res = torch.stack(neighborhood_list, dim=2)
        return vector_res

    def bilinear_interpolate(self, vec):
        vec = vec.clone()  # Avoid modifying the input directly
        vec[:, 0] = vec[:, 0] / math.pi * (self.grid_size[0] - 1)
        vec[:, 1] = (vec[:, 1] + math.pi) / (2 * math.pi) * (self.grid_size[1] - 1)

        W, H = self.grid_size[1] - 1, self.grid_size[0] - 1
        top_left_x = vec[:, 0].long().to(self.device)
        top_left_y = vec[:, 1].long().to(self.device)

        x_fract = vec[:, 0] % 1
        y_fract = vec[:, 1] % 1

        bottom_right_x = torch.where(
            top_left_x + 1 > W, torch.tensor(0, device=self.device), top_left_x + 1
        )
        bottom_right_y = torch.where(
            top_left_y + 1 > H, torch.tensor(0, device=self.device), top_left_y + 1
        )

        tl_vectors = self.grid_pos[top_left_y, top_left_x]  # Top-left corner vectors
        tr_vectors = self.grid_pos[
            top_left_y, bottom_right_x
        ]  # Top-right corner vectors
        bl_vectors = self.grid_pos[
            bottom_right_y, top_left_x
        ]  # Bottom-left corner vectors
        br_vectors = self.grid_pos[
            bottom_right_y, bottom_right_x
        ]  # Bottom-right corner vectors
        top_interp = (1 - x_fract).unsqueeze(1) * tl_vectors + x_fract.unsqueeze(
            1
        ) * tr_vectors
        bottom_interp = (1 - x_fract).unsqueeze(1) * bl_vectors + x_fract.unsqueeze(
            1
        ) * br_vectors
        interpolated_vectors = (1 - y_fract).unsqueeze(
            1
        ) * top_interp + y_fract.unsqueeze(1) * bottom_interp

        return interpolated_vectors

    def forward(self, pos):
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
        # if dir.dim() == 1:
        #    dir = dir.unqueeeze(1)
        # x = torch.cat((pos, dir), dim=1)
        pos_weight = self.dirCNN(pos)
        pos_neighbor = self.get_neighbor(pos)
        x = pos_weight*pos_neighbor
        x = x.sum(dim=[2, 3]) 
        # x = self.bilinear_interpolate(pos)
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
