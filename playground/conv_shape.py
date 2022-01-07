import torch
a = torch.randn([600, 1, 28, 28])
conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(1,8,3,1,1),
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(4)
            )
dense = torch.nn.Sequential(
    torch.nn.Linear(392, 10),
    torch.nn.ReLU()
)
b = conv1(a)
b = b.view(b.size(0), -1)
out = dense(b)
print(b.size())