import torch
from networks.M.nethit import BnHitNet
from RenderDataset import RenderDatasetM
from torch.utils.data import DataLoader
import torch.nn as nn
import wandb
import torch.nn.init as init

need_wandb = True
def init_model(m):
    if isinstance(m, nn.Linear):
        init.uniform_(m.weight, a=-1e-9, b=1e-9)
        init.uniform_(m.bias, a=-1e-9, b=1e-9)

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.bce_loss = nn.BCELoss()  # Binary Cross-Entropy loss
        self.mse_loss = nn.SmoothL1Loss()  # Mean Squared Error loss
        
    def forward(self, hit_miss_pred, rgb_pred, hit_miss_true, rgb_true):
        # Compute the BCE loss for hit/miss classification
        if hit_miss_pred.dim() == 1:
            hit_miss_pred = hit_miss_pred.unsqueeze(1)
        if rgb_pred.dim() == 1:
            rgb_pred = rgb_pred.unsqueeze(1)
        if hit_miss_true.dim() == 1:
            hit_miss_true = hit_miss_true.unsqueeze(1)
        if rgb_true.dim()==1:
            rgb_true = rgb_true.unsqueeze(1)
        classification_loss = self.bce_loss(hit_miss_pred, hit_miss_true)
        rgb_true = rgb_true
        # Compute the MSE loss for RGB prediction, only for hit samples
        # Mask to select only the hit cases
        hit_mask = hit_miss_true > 0.5
        hit_mask = hit_mask.squeeze()
        rgb_loss = self.mse_loss(rgb_pred[hit_mask,:], rgb_true[hit_mask,:])
        
        # Combine the two losses
        total_loss = classification_loss + rgb_loss
        return self.mse_loss(rgb_pred,rgb_true)
    
def train(model_path,dataset_path,echo):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BnHitNet(L=6,device=device)
    model.to(device)
    model.apply(init_model)
    dataset = RenderDatasetM(data_dir=dataset_path)
    dataset_loader = DataLoader(dataset,batch_size=20000,shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    total_step = len(dataset_loader)
    criterion = CustomLoss()
    for epoch in range(echo):
        for i, (para, labels,hit) in enumerate(dataset_loader):  
            # Move tensors to the configured device
            para = para.to(device)
            labels = labels.to(device)
            hit = hit.to(device)
            p1,p2 = torch.split(para,2,dim=1)
            hit_p,rgb_p = model(p1)
            loss = criterion(hit_p, rgb_p,hit,labels)
            if need_wandb:
                wandb.log({"loss":loss})
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # for param in model.parameters():
            #     if param.grad is not None:
            #         print(param.grad.data.abs().mean())

            if (i+1) % 50 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, echo, i+1, total_step, loss.item()))
    if need_wandb:
        wandb.finish()
    model.eval()
    example_input = torch.rand(2,2).to(device)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(model_path)
if __name__ == "__main__":
    if need_wandb:
        wandb.init(
        # set the wandb project where this run will be logged
        project="Neural_Rendering",

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.1,
        "epochs": 100,
        "model":"res",
        "feature_num":0,
        "train_type":"sh"
        }
    )
    # save path
    # train(model_type="grid",model_path="models/model_sph_grid7.pt",dataset_path="datas/sph_6.json",echo=100)
    train(model_path="models/sh1.pt",dataset_path="datas/M/sph1.json",echo=3)