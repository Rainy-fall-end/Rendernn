import torch
from simpleNet import NeuralNetwork
from RenderDataset import RenderDateset
from torch.utils.data import DataLoader
import torch.nn as nn
def train():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork(6,3).to(device)
    dataset = RenderDateset(data_dir="c.json")
    dataset_loader = DataLoader(dataset,batch_size=100,shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0004)
    # Train the model
    total_step = len(dataset_loader)
    for epoch in range(10):
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
                    .format(epoch+1, 100, i+1, total_step, loss.item()))
    model.eval()
    example_input = torch.rand(2,6).to(device)
    input = torch.tensor([209.240677, 51.0370140, -13.8729544, -0.742403865, -0.519836783, -0.422618270],dtype=torch.float32).to(device)
    print(model(input))
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save("model2.pt")
if __name__ == "__main__":
    train()