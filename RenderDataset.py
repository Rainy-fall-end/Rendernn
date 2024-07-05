import torch
from torch.utils.data import Dataset
import json
from torch import nn
from torch.nn import functional as F
class RenderDataset(Dataset):
    def __init__(self,data_dir="datas/a.json",transform=None) -> None:
        super().__init__()
        with open(data_dir) as f:
            self.datas = json.load(f)
        self.transform = transform
    def __getitem__(self, index):
        para = torch.Tensor(self.datas[index]["point"] + self.datas[index]["dir"])
        label = torch.Tensor(self.datas[index]["rgb"])
        if self.transform is not None:
            para = self.transform(para)
            label = self.transform(label)
        return para,label
    def __len__(self)->int:
        return len(self.datas)

class RenderDatasetSph(Dataset):
    def __init__(self,data_dir="datas/sph_1.json",transform=None) -> None:
        super().__init__()
        with open(data_dir) as f:
            self.datas = json.load(f)
        self.transform = transform
    def __getitem__(self, index):
        para = torch.Tensor(self.datas[index]["point_sph"]+self.datas[index]["dir_sph"])
        label = torch.Tensor(self.datas[index]["rgb"])
        if self.transform is not None:
            para = self.transform(para)
            label = self.transform(label)
        return para,label
    def __len__(self)->int:
        return len(self.datas)
class RenderDatasetB(Dataset):
    def __init__(self,data_dir="datas/sph_1.json",transform=None) -> None:
        super().__init__()
        with open(data_dir) as f:
            self.datas = json.load(f)
        self.transform = transform
    def __getitem__(self, index):
        para = torch.Tensor(self.datas[index]["point_sph"]+self.datas[index]["dir_sph"])
        label = torch.Tensor(self.datas[index]["rgb"])
        if(torch.allclose(label, torch.tensor([0,0,1],dtype=torch.float32))):
            label = torch.Tensor([0])
        else:
            label = torch.Tensor([1])
        if self.transform is not None:
            para = self.transform(para)
            label = self.transform(label)
        return para,label
    def __len__(self)->int:
        return len(self.datas)
## test

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = RenderDataset()
    dataset_loader = DataLoader(dataset,batch_size=4,shuffle=True)
    for batch in dataset_loader:
        para,label = batch