import torchvision
import torch

def googlenet():
    cnn_neural_network =  torchvision.models.googlenet(pretrained=True)
    cnn_neural_network.fc = torch.nn.Linear(in_features=1024, out_features=3, bias=True)
    return(cnn_neural_network)

if __name__ == "__main__":
    model = googlenet()
