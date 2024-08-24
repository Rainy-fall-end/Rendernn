import json
import numpy as np
import pdb
import torch

from networks.ray_utils import get_rays, get_ray_directions, get_ndc_rays


def shash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result


def get_bbox3d_for_blenderobj(camera_transforms, H, W, near=2.0, far=6.0):
    camera_angle_x = float(camera_transforms['camera_angle_x'])
    focal = 0.5*W/np.tan(0.5 * camera_angle_x)

    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    points = []

    for frame in camera_transforms["frames"]:
        c2w = torch.FloatTensor(frame["transform_matrix"])
        rays_o, rays_d = get_rays(directions, c2w)
        
        def find_min_max(pt):
            for i in range(3):
                if(min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if(max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W-1, H*W-W, H*W-1]:
            min_point = rays_o[i] + near*rays_d[i]
            max_point = rays_o[i] + far*rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

    return (torch.tensor(min_bound)-torch.tensor([1.0,1.0,1.0]), torch.tensor(max_bound)+torch.tensor([1.0,1.0,1.0]))


def get_bbox3d_for_llff(poses, hwf, near=0.0, far=1.0):
    H, W, focal = hwf
    H, W = int(H), int(W)
    
    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    points = []
    poses = torch.FloatTensor(poses)
    for pose in poses:
        rays_o, rays_d = get_rays(directions, pose)
        rays_o, rays_d = get_ndc_rays(H, W, focal, 1.0, rays_o, rays_d)

        def find_min_max(pt):
            for i in range(3):
                if(min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if(max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W-1, H*W-W, H*W-1]:
            min_point = rays_o[i] + near*rays_d[i]
            max_point = rays_o[i] + far*rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

    return (torch.tensor(min_bound)-torch.tensor([0.1,0.1,0.0001]), torch.tensor(max_bound)+torch.tensor([0.1,0.1,0.0001]))


def get_voxel_vertices(pd, box_min,box_max,box_offset, resolution, log2_hashmap_size):
    '''
    pd: 3D positions and directions of samples. B x 4
    bounding_box: min and max pd coordinates of object bbox
    resolution: number of voxels per axis
    '''

    keep_mask = pd==torch.max(torch.min(pd, box_max), box_min)
    if not torch.all(pd <= box_max) or not torch.all(pd >= box_min):
        print("ALERT: some points are outside bounding box. Clipping them!")
        pd = torch.clamp(pd, min=box_min, max=box_max)

    grid_size = (box_max-box_min)/resolution
    
    bottom_left_idx = torch.floor((pd-box_min)/grid_size).int()
    voxel_min_vertex = bottom_left_idx*grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0,1.0],device='cuda')*grid_size
    
    # BOX_OFFSETS = torch.tensor(
    #     [[[i, j, k, l] for i in [0, 1] for j in [0, 1] for k in [0, 1] for l in [0, 1]]],
    #     device='cuda'
    # )
    voxel_indices = bottom_left_idx.unsqueeze(1) + box_offset
    
    hashed_voxel_indices = shash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask



if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test = random_tensor = torch.rand(2,8, 3).to(device=device)
    res = shash(test,19)
    print(res.shape)
