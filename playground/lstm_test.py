import torch
# import matplotlib.pyplot as plt
import numpy as np

input_feaure = 4
input_length = 10
hidden_size = 128
stride = 5

class SeriesTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_feaure, hidden_size=hidden_size, num_layers=input_length,
                                  batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, input_feaure)

    def forward(self, x):
        h0 = torch.zeros(input_length, x.size(0), hidden_size).requires_grad_()
        c0 = torch.zeros(input_length, x.size(0), hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


a = torch.linspace(0, 2000, steps=2000)
a = torch.sqrt(a) * (torch.sin(a/100) + 1.5)
print(a.size())

idx = 0
X = []
y = []
while idx + input_length < len(a):
    X.append(a[idx: idx+input_length])
    y.append(a[idx+input_length])
    idx += stride
X = torch.vstack(X)
y = torch.vstack(y)
X = X.unsqueeze(dim=1)
# print(X.size())
X = torch.cat(input_feaure * [X], dim=1)
X = X.transpose(1, 2)
X = X + torch.randn(X.size()) * 0.06
y = torch.hstack(input_feaure * [y])
print(X.size(), y.size())
# print(X[1], y[1])

m = SeriesTest()
opt = torch.optim.Adam(m.parameters())
loss_fn = torch.nn.MSELoss()
for i in range(100):
    out = m(X)
    loss = loss_fn(out, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    pred_y = torch.mean(out, dim=1)
    truth_y = torch.mean(y, dim=1)
    diff = torch.mean(pred_y-truth_y)
    print(f'Epoch {i} - avg diff = {diff}')
