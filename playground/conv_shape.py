import torch

reso = 28
a = torch.randn([600, 1, reso, reso])
pv = 4
conv1 = torch.nn.Sequential(
    torch.nn.Conv2d(pv**2,8,3,1,1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2)
)
dense = torch.nn.Sequential(
    torch.nn.Linear(((reso//pv)//2)**2 * 8, 10),
    torch.nn.ReLU()
)
a = a.view(a.size(0), pv**2, a.size(2)//pv, -1)
b = conv1(a)
b = b.view(b.size(0), -1)
out = dense(b)
print(out.size())