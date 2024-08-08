import torch
from networks.simpleNet import NeuralNetwork
from networks.resnet import res_net
from networks.gridnet3 import GridNet
from networks.hashnet import HashNet
from RenderDataset import RenderDataset,RenderDatasetSph,RenderDatasetB
from torch.utils.data import DataLoader
import torch.nn as nn
import wandb
import torch.nn.init as init
import numpy

need_wandb = True


def init_model(m):
    if isinstance(m, nn.Linear):
        init.uniform_(m.weight, a=-1e-9, b=1e-9)
        init.uniform_(m.bias, a=-1e-9, b=1e-9)

def train(model_type,model_path,dataset_path,echo):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == "grid":
        model = GridNet([512,512],5,3,device).to(device)    
    elif model_type == "res":
        model = res_net(2,3).to(device)
    elif model_type == "hash":
        model = HashNet(bounding_box=torch.tensor([[0,0,0],[1,1,1]]).to(device=device))
    else:
        model = NeuralNetwork(2,3).to(device)
    model.apply(init_model)
    dataset = RenderDatasetSph(data_dir=dataset_path)
    dataset_loader = DataLoader(dataset,batch_size=10000,shuffle=True)
    criterion = nn.MSELoss()
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
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
                p1,p2 = torch.split(para,2,dim=1)
                outputs = model(p1)
            elif(model_type=="sh"):
                p1,p2 = torch.split(para,2,dim=1)
                outputs = model(p1)
            else:
                p1,p2 = torch.split(para,2,dim=1)
                outputs = model(p1)
                
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
    # numpy_array = model.grid.cpu().detach().numpy()
    # numpy.savetxt("b.txt",numpy_array[:,:,0],fmt='%.2f')
    model.eval()
    if(model_type=="grid"):
        example_input_1 = torch.rand(2,2).to(device)
        example_input_2 = torch.rand(2,2).to(device)
        traced_script_module = torch.jit.trace(model, example_input_1)
    else:
        input1 = torch.tensor([1.85491669, 0.721287251],device=device)
        print(model(input1))
        input2 = torch.tensor([1.85491669, 0.739502788],device=device) #, 2.00712872, -2.53072739
        print(model(input2))
        input3 = torch.tensor([1.65491669, 0.739502788],device=device) #, 2.00712872, -2.53072739
        print(model(input3))
        input1 = torch.tensor([1.697962648764957, 1.7065286256265528],device=device)
        print(model(input1))
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
        "train_type":"sh",
        "name" : "grid5X5"
        }
    )
    # save path
    # train(model_type="grid",model_path="models/model_sph_grid7.pt",dataset_path="datas/sph_6.json",echo=100)
    train(model_type="grid",model_path="models/sh3.pt",dataset_path="datas/sph_6.json",echo=5)