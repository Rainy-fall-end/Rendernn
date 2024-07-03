import torch
from simpleNet import NeuralNetwork
from RenderDataset import RenderDateset,RenderDatasetSph
from torch.utils.data import DataLoader
import torch.nn as nn
def train():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork(4,3).to(device)
    dataset = RenderDatasetSph(data_dir="datas/sph_3.json")
    dataset_loader = DataLoader(dataset,batch_size=100,shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0.004)
    # Train the model
    total_step = len(dataset_loader)
    for epoch in range(100):
        for i, (para, labels) in enumerate(dataset_loader):  
            # Move tensors to the configured device
            para = para.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(para)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, 300, i+1, total_step, loss.item()))
    model.eval()
    example_input = torch.rand(2,4).to(device)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save("models/model_sph3.pt")
if __name__ == "__main__":
    train()