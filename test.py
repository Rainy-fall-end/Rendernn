import torch
from simpleNet import NeuralNetwork
# model = NeuralNetwork(6,3)
# model.load_state_dict(torch.jit.load("model.pt"))
model = torch.jit.load("model2.pt")
model.eval()
model.to("cpu")
input = torch.tensor([209.725220, 50.3450203, -13.8729544, -0.742403865, -0.519836783, -0.422618270])
res = model(input)
input1 = torch.tensor([232.983185,17.1292095,-13.8729544,-0.742403865,-0.519836783,-0.422618270])
res_ = model(input1)
print(torch.version.cuda)
print(torch.__version__)
