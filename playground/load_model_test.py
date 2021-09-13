import torch
import torchvision
import pandas as pd
import numpy as np

train_data = torchvision.datasets.CIFAR10("./datasets", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("./datasets", train=False, transform=torchvision.transforms.ToTensor(), download=True)

model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
print(model)
for param_tensor in model.state_dict():
    print(param_tensor, '\t', model.state_dict()[param_tensor].size())