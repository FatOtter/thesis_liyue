import torch
import torchvision
import torch.utils.data.dataloader as Data
import pandas as pd
from datetime import datetime

NUM_OF_PARTICIPANTS = 10
NUM_OF_EPOCH = 5
BATCH_SIZE = 32
RECORDING_PATH = "../trainig_records/"

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.5,],  # mean=[0.5071, 0.4865, 0.4409] for cifar100
        std=[0.5,])
])

train_data = torchvision.datasets.MNIST("./mnist", True, transform, download=True)
test_data = torchvision.datasets.MNIST("./mnist", False, transform)

print("train_data:", train_data.train_data.size())
print("train_labels:", train_data.train_labels.size())
print("test_data:", test_data.test_data.size())

class DataSplitter:
    def __init__(self, participants = NUM_OF_PARTICIPANTS, balanced = True):
        self.number_of_participants = participants
        self.balanced = balanced

    def split(self, train_data, test_data):
        train_per_participant = train_data.train_data.size()[0]//self.number_of_participants
        test_per_participant = test_data.test_data.size()[0]//self.number_of_participants
        train_split_size = []
        test_split_size = []
        for i in range(self.number_of_participants):
            if i < self.number_of_participants - 1:
                train_split_size.append(train_per_participant)
                test_split_size.append(test_per_participant)
            else:
                train_split_size.append(train_data.train_data.size()[0] - (self.number_of_participants - 1)*train_per_participant)
                test_split_size.append(test_data.test_data.size()[0] - (self.number_of_participants - 1)*test_per_participant)
        print(train_split_size, test_split_size)
        train = torch.utils.data.random_split(train_data, train_split_size)
        test = torch.utils.data.random_split(test_data, test_split_size)
        print(len(train))
        return train, test


splitter = DataSplitter(NUM_OF_PARTICIPANTS)
train, test = splitter.split(train_data, test_data)


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


models = {}
optimizers = {}
train_loaders = {}
test_loaders = {}
for i in range(NUM_OF_PARTICIPANTS):
    models[i] = Model1()
    optimizers[i] = torch.optim.Adam(models[i].parameters())
    train_loaders[i] = Data.DataLoader(train[i], batch_size=BATCH_SIZE, shuffle=True)
    test_loaders[i] = Data.DataLoader(test[i], batch_size=BATCH_SIZE)
loss_func = torch.nn.CrossEntropyLoss()
records = pd.DataFrame(columns=['epoch', 'participant', 'training_loss', 'training_acc', 'test_loss', 'test_acc'])

for epoch in range(NUM_OF_EPOCH):
    print("Epoch {}:".format(epoch))
    train_loss = torch.zeros(NUM_OF_PARTICIPANTS)
    train_acc = torch.zeros(NUM_OF_PARTICIPANTS)
    for participant in range(NUM_OF_PARTICIPANTS):
        print("Training for participant {} ...".format(participant))
        # print(models[participant])
        for batch_x, batch_y in train_loaders[participant]:
            #print(batch_x.type())
            out = models[participant](batch_x)
            loss = loss_func(out, batch_y)
            train_loss[participant] += loss.item()
            pred = torch.max(out, 1).indices
            train_correct = (pred == batch_y).sum()
            train_acc[participant] += train_correct.item()
            optimizers[participant].zero_grad()
            loss.backward()
            optimizers[participant].step()
        train_loss_temp, train_acc_temp = train_loss[participant] / len(train[participant]), train_acc[participant] / len(train[participant])
        print("Training complete for participant {}, train loss {}, train accuracy {}".format(participant, train_loss_temp, train_acc_temp))

        models[participant].eval()
        eval_loss = 0
        eval_acc = 0
        for batch_x, batch_y in test_loaders[participant]:
            with torch.no_grad():
                out = models[participant](batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1).indices
            test_correct = (pred == batch_y).sum()
            eval_acc += test_correct.item()
        eval_loss, eval_acc = eval_loss / len(test[participant]), eval_acc / len(test[participant])
        print('Test loss: {}, Test Acc: {}'.format(eval_loss, eval_acc))
        records.loc[len(records)] = [epoch, participant, train_loss_temp.item(), train_acc_temp.item(), eval_loss, eval_acc]

now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H")
records.to_csv(RECORDING_PATH+time_str+".csv")
