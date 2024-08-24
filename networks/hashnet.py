import torch

# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import torch.nn.init as init
from networks.utils import get_voxel_vertices

class DirCNN(nn.Module):
    def __init__(self):
        super(DirCNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2**4)
        )
    def forward(self, x):
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

class HashEmbedder(nn.Module):
    def __init__(
        self,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        finest_resolution=512,
    ):
        super(HashEmbedder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.box_min = torch.tensor([0, -torch.pi,0.5*torch.pi, -0.85*torch.pi]).to(self.device)
        self.box_max = torch.tensor([torch.pi, torch.pi,0.85*torch.pi, -0.5*torch.pi]).to(
            self.device
        )
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level

        self.b = torch.exp(
            (torch.log(self.finest_resolution) - torch.log(self.base_resolution))
            / (n_levels - 1)
        )

        num_embeddings = 2 ** self.log2_hashmap_size

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings, self.n_features_per_level)
                for i in range(n_levels)
            ]
        )

        self.box_offset = torch.tensor(
            [
                [
                    [i, j, k, l]
                    for i in [0, 1]
                    for j in [0, 1]
                    for k in [0, 1]
                    for l in [0, 1]
                ]
            ],
            device="cuda",
        )
        self.weightNet = DirCNN().to(self.device)
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()

    def quadlinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        """
        x: B x 4
        voxel_min_vertex: B x 4
        voxel_max_vertex: B x 4
        voxel_embedds: B x 16 x 2  # 4D cube has 16 vertices
        """
        # weights now computed for 4 dimensions
        weights = (x - voxel_min_vertex) / (
            voxel_max_vertex - voxel_min_vertex
        )  # B x 4

        # step 1: interpolate along the 1st dimension (x)
        c0000 = (
            voxel_embedds[:, 0] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 8] * weights[:, 0][:, None]
        )
        c0001 = (
            voxel_embedds[:, 1] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 9] * weights[:, 0][:, None]
        )
        c0010 = (
            voxel_embedds[:, 2] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 10] * weights[:, 0][:, None]
        )
        c0011 = (
            voxel_embedds[:, 3] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 11] * weights[:, 0][:, None]
        )
        c0100 = (
            voxel_embedds[:, 4] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 12] * weights[:, 0][:, None]
        )
        c0101 = (
            voxel_embedds[:, 5] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 13] * weights[:, 0][:, None]
        )
        c0110 = (
            voxel_embedds[:, 6] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 14] * weights[:, 0][:, None]
        )
        c0111 = (
            voxel_embedds[:, 7] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 15] * weights[:, 0][:, None]
        )

        # step 2: interpolate along the 2nd dimension (y)
        c000 = c0000 * (1 - weights[:, 1][:, None]) + c0100 * weights[:, 1][:, None]
        c001 = c0001 * (1 - weights[:, 1][:, None]) + c0101 * weights[:, 1][:, None]
        c010 = c0010 * (1 - weights[:, 1][:, None]) + c0110 * weights[:, 1][:, None]
        c011 = c0011 * (1 - weights[:, 1][:, None]) + c0111 * weights[:, 1][:, None]

        # step 3: interpolate along the 3rd dimension (z)
        c00 = c000 * (1 - weights[:, 2][:, None]) + c010 * weights[:, 2][:, None]
        c01 = c001 * (1 - weights[:, 2][:, None]) + c011 * weights[:, 2][:, None]

        # step 4: interpolate along the 4th dimension (w)
        c = c00 * (1 - weights[:, 3][:, None]) + c01 * weights[:, 3][:, None]

        return c

    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []
        
        log2_hashmap_size_tensor = torch.tensor(self.log2_hashmap_size, device=self.device)

        for i, embedding in enumerate(self.embeddings):
            resolution = torch.floor(self.base_resolution * self.b**i).to(self.device)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = (
                get_voxel_vertices(
                    x, self.box_min, self.box_max, self.box_offset, resolution, log2_hashmap_size_tensor
                )
            )

            voxel_embedds = embedding(hashed_voxel_indices)
            resolution_n = resolution.reshape(-1, 1).expand(x.size(0), 1)
            resolution_x = torch.cat((resolution_n,x),dim=1)
            weight = self.weightNet(resolution_x)
            # x_embedded = self.quadlinear_interp(
            #     x, voxel_min_vertex, voxel_max_vertex, voxel_embedds
            # )
            x_embedded = (weight.unsqueeze(2)*voxel_embedds).sum(dim=1)
            x_embedded_all.append(x_embedded)


        keep_mask = keep_mask.sum(dim=-1) == keep_mask.shape[-1]
        return torch.cat(x_embedded_all, dim=-1), keep_mask


class HashNet(nn.Module):
    def __init__(
        self,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        finest_resolution=512,
    ) -> None:
        super(HashNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embe = HashEmbedder(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
        )
        self.skip = [4]
        # self.output_layers = nn.ModuleList(
        #     [nn.Linear(n_levels * n_features_per_level, 3)]
        # )
        self.output_layers = nn.Sequential(
            nn.Linear(n_levels * n_features_per_level, 1)
        )
    def forward(self, x):
        x_ori = x.clone()
        x = self.embe(x)[0]
        # x = torch.cat((x_ori,x),dim=-1)
        # for i, l in enumerate(self.output_layers):
        #     x = l(x)
        #     x = F.leaky_relu(x)
        #     if i in self.skip:
        #         x = torch.cat([x, x_embe], dim=-1)
        x = self.output_layers(x)
        x = torch.sigmoid(x)
        # x = F.threshold(x, 0.1, 0.0)
        # x = F.threshold(x, 0.0, 255.0)
        # x = x * 255
        return x



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = random_tensor = torch.rand(2, 4).to(device=device)
    bounding_box = torch.tensor(
        [[0, -torch.pi, 0, -torch.pi], [torch.pi, torch.pi, torch.pi, torch.pi]]
    ).to(device=device)
    a, b = bounding_box
    # model = HashEmbedder(bounding_box=bounding_box,
    #                      n_levels=16,
    #                      n_features_per_level=2).to(device=device)
    model = HashNet(bounding_box=bounding_box).to(device=device)
    res = model(test)
    print(res.shape)
