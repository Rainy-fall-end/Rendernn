import torch
from networks.simpleNet import NeuralNetwork
from networks.resnet import res_net
from networks.gridnet2 import GridNetDir
from RenderDataset import RenderDataset,RenderDatasetSph,RenderDatasetB
from torch.utils.data import DataLoader
import torch.nn as nn
import wandb
import torch.nn.init as init
import math
need_wandb = True
def init_model(m):
    if isinstance(m, nn.Linear):
        init.uniform_(m.weight, a=-1e-9, b=1e-9)
        init.uniform_(m.bias, a=-1e-9, b=1e-9)

def train(model_type,model_path,dataset_path,echo):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == "grid":
        model = GridNetDir([50,50,25,25],6,device).to(device)    
    elif model_type == "res":
        model = res_net(4,3).to(device)
    else:
        model = NeuralNetwork(4,3).to(device)
    model.apply(init_model)
    dataset = RenderDatasetSph(data_dir=dataset_path)
    dataset_loader = DataLoader(dataset,batch_size=512,shuffle=True)
    criterion = nn.MSELoss()
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.004)
    # Train the model
    total_step = len(dataset_loader)
    for epoch in range(echo):
        for i, (para, labels) in enumerate(dataset_loader):  
            # Move tensors to the configured device
            para = para.to(device)
            labels = labels.to(device)
            
            # Forward pass
            if(model_type=="grid"):
                outputs = model(para)
            else:
                outputs = model(para)
            loss = criterion(outputs, labels)
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
    if(model_type=="grid"):
        test = torch.tensor([0.8*math.pi,0.8*math.pi,2,-2.53]).to(device)
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, model_path)
    else:
        input1 = torch.tensor([[0.8*math.pi,0.8*math.pi],[0.3,0.4]],device=device)
        traced_script_module = torch.jit.trace(model, input1)
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
        "train_type":"grid-dir"
        }
    )
    train(model_type="grid",model_path="models/dir_model_sph_grid4.pt",dataset_path="datas/all_dir_sph_range_2.json",echo=1)