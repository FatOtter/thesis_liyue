import torch
import torchvision


class participant(torch.nn.Module):
    def __init__(self):
        super(participant, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(4)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(392, 10),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z1 = self.conv1(x)
        # z1 = z1.view(z1.size(0), -1)
        out = self.dense(z1)
        return out


MNIST_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.5, ],
                std=[0.5, ])
        ])
DATA_SET_PATH = "../datasets"
train_set = torchvision.datasets.MNIST(DATA_SET_PATH, True, MNIST_transform, download=True)
test_set = torchvision.datasets.MNIST(DATA_SET_PATH, False, MNIST_transform)
