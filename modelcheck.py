import pdb
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

cnn_neural_network = torchvision.models.googlenet()
#print(cnn_neural_network)
#pdb trace, print cnnneuralnetwork, look for the infeatures, change it and run the code again without the pdb

cnn_neural_network.fc = torch.nn.Linear(in_features=1024, out_features=3, bias=True)
print(count_parameters(cnn_neural_network))

# mobilenet_v2 = models.mobilenet_v2()
#mobilenet_v3_large = models.mobilenet_v3_large()
#mobilenet_v3_small = models.mobilenet_v3_small()
