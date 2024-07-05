import torch
from networks.simpleNet import NeuralNetwork
# model = NeuralNetwork(6,3)
# model.load_state_dict(torch.jit.load("model.pt"))
model = torch.jit.load("models/model_sph12.pt")
model.eval()
model.to("cpu")
input = torch.tensor([1.85491669, 0.721287251])
print(model(input))
input1 = torch.tensor([1.85491669, 0.739502788]) #, 2.00712872, -2.53072739
print(model(input1))
print(torch.version.cuda)
print(torch.__version__)
