from datetime import datetime

import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import torch.utils.data.dataloader as Data
import pandas as pd

RECORDING_PATH = "./trainig_records/"

train_data = torchvision.datasets.MNIST("./mnist", True, torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST("./mnist", False, torchvision.transforms.ToTensor())

print("train_data:", train_data.train_data.size())
print("train_labels:", train_data.train_labels.size())
print("test_data:", test_data.test_data.size())
#print(train_data.train_data[0])

train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=64)

class Model1(torch.nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,8,3,1,1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(8,16, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(16*3*3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out

model = Model1()
for parameter in model.parameters():
    print(len(parameter))

optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()

df = pd.DataFrame()
df2 = pd.DataFrame(columns=["Training loss", "Training Acc", "Test loss", "Test Acc"])

for epoch in range(5):
    print("Epoch {}:".format(epoch+1))
    train_loss = 0
    train_acc = 0
    batch_counter = 0
    overall_batches = len(train_data)/64
    for batch_x, batch_y in train_loader:
        batch_counter+=1
        if(batch_counter%100 == 0):
            print("Currently {}/{} batches has trained".format(batch_counter, overall_batches))
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        #print(batch_x[0])
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.item()
        pred = torch.max(out, 1).indices
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train loss: {}, Train Acc: {}'.format(train_loss/len(train_data), train_acc/len(train_data)))


    # Evaluation
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        with torch.no_grad():
            out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1).indices
        test_correct = (pred == batch_y).sum()
        eval_acc += test_correct.item()
    print('Test loss: {}, Test Acc: {}'.format(eval_loss/len(test_data), eval_acc/len(test_data)))

    # recording parameters
    df2.loc[epoch] = [train_loss/len(train_data), train_acc/len(train_data), eval_loss/len(test_data), eval_acc/len(test_data)]

    all_param = torch.empty(1)
    for param in model.parameters():
        param = param.flatten()
        all_param = torch.cat([all_param, param])
    df['epoch{}'.format(epoch)] = all_param.detach().numpy()

now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H")
df.to_csv(RECORDING_PATH+"Params"+time_str+".csv")
df2.to_csv(RECORDING_PATH+"Outcome"+time_str+".csv")
