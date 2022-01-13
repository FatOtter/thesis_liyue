import numpy as np
import torch

# test_raw = np.genfromtxt("../datasets/pretrained_cifar10/cifar100_resnet56_test.csv", delimiter=',')
# print(test_raw[0])
# test_label = np.array(test_raw[:, -1], dtype=np.long)
# test_label_one_hot = np.zeros((test_label.size, test_label.max()+1))
# test_label_one_hot[np.arange(test_label.size), test_label] = 1
# test_data = np.array(test_raw[:, :-1], dtype=np.float)
# train_raw = np.genfromtxt("../datasets/pretrained_cifar10/cifar100_resnet56_train.csv", delimiter=',')
# print(train_raw[0])
# train_label = np.array(train_raw[:, -1], dtype=np.long)
# train_label_one_hot = np.zeros((train_label.size, train_label.max()+1))
# train_label_one_hot[np.arange(train_label.size), train_label] = 1
# train_data = np.array(train_raw[:, :-1], dtype=np.float)
# np.savez_compressed("cifar_resnet56", train_data, train_label, train_label_one_hot, test_data, test_label, test_label_one_hot)


loaded = np.load("cifar_resnet56.npz")
train_data = loaded['arr_0']
train_label = loaded['arr_1']
train_label_one_hot = loaded['arr_2']
test_data = loaded['arr_3']
test_label = loaded['arr_4']

class model(torch.nn.Module):
    def __init__(self, n_H):
        super(model, self).__init__()
        self.input = torch.nn.Sequential(
            torch.nn.Linear(64, n_H),
            torch.nn.ReLU()
        )
        self.output = torch.nn.Sequential(
            torch.nn.Linear(n_H, 10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.input(x)
        return self.output(out)

batch = 100
b_size = 50000 // batch
m = model(512)
loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.RMSprop(m.parameters())

train_x = torch.tensor(train_data, dtype=torch.float)
train_y = torch.tensor(train_label, dtype=torch.long)
test_x = torch.tensor(test_data, dtype=torch.float)
test_y = torch.tensor(test_label, dtype=torch.long)
for i in range(100):
    acc = 0
    for j in range(batch):
        lower = j * b_size
        upper = lower + b_size
        batch_x = train_x[lower: upper]
        batch_y = train_y[lower: upper]
        # batch_y = batch_y.flatten()
        optim.zero_grad()
        out = m(batch_x)
        pred_y = torch.max(out, dim=1).indices
        loss = loss_fn(out, batch_y)
        loss.backward()
        optim.step()
        b_acc = (pred_y == batch_y).sum()
        acc += b_acc
    acc = acc/50000
    with torch.no_grad():
        out = m(test_x)
    pred_y = torch.max(out, dim=1).indices
    test_acc = (pred_y == test_y).sum()
    test_acc = test_acc / 10000
    print(f'Epoch {i} - train acc {acc:6.4f}, test acc {test_acc:6.4f}')
