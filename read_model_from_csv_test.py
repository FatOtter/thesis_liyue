import torch
import pandas as pd
import torchvision
from torch.utils.data.dataloader import DataLoader
from model_shallow_cnn import ShallowCNN

EPOCH = 0

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.5,],  # mean=[0.5071, 0.4865, 0.4409] for cifar100
        std=[0.5,])
])

train_data = torchvision.datasets.MNIST("./mnist", True, transform, download=True)
test_data = torchvision.datasets.MNIST("./mnist", False, transform)
loss_func = torch.nn.CrossEntropyLoss()

RECORDING_PATH = "./trainig_records/"
df = pd.read_csv(RECORDING_PATH+"anchor.csv")


class ModelLoader:
    def __init__(self):
        self.model = ShallowCNN()
        self.column_name = ""
        self.data_frame = pd.DataFrame()

    def load(self, path, column_name, start_index=1):
        self.data_frame = pd.read_csv(path)
        for param in self.model.parameters():
            length = len(param.flatten())
            loaded = self.data_frame[column_name][start_index:start_index+length]
            loaded = loaded.to_numpy()
            loaded_tensor = torch.tensor(loaded)
            loaded_tensor = loaded_tensor.reshape(param.size())
            with torch.no_grad():
                param.copy_(loaded_tensor)
            start_index += length
        return self.model


# model = ShallowCNN()
# current_index = 1
# for param in model.parameters():
#     length = len(param.flatten())
#     loaded = df['epoch{}'.format(EPOCH)][current_index:current_index+length]
#     loaded = loaded.to_numpy()
#     loaded_tensor = torch.tensor(loaded)
#     loaded_tensor = loaded_tensor.reshape(param.size())
#     # print(param.size())
#     # print(loaded_tensor.size())
#     with torch.no_grad():
#         param.copy_(loaded_tensor)
#     current_index += length
# print(current_index, len(df['epoch1']))

loader = ModelLoader()
model = loader.load(RECORDING_PATH+"anchor.csv", 'epoch{}'.format(EPOCH))

loss_accumulator = 0
acc_accumulator = 0
for batch_x, batch_y in DataLoader(test_data, batch_size=64):
    with torch.no_grad():
        out = model(batch_x)
    loss = loss_func(out, batch_y)
    loss_accumulator += loss.item()
    pred = torch.max(out, 1).indices
    test_correct = (pred == batch_y).sum()
    acc_accumulator += test_correct.item()
print("Test loss: {}, test accuracy: {} ".format(loss_accumulator/len(test_data), acc_accumulator/len(test_data)))

# new_df = pd.DataFrame()
# all_param = torch.empty(1)
# for param in model.parameters():
#     param = param.flatten()
#     all_param = torch.cat([all_param, param])
# new_df['epoch{}'.format(EPOCH)] = all_param.detach().numpy()
# comp_accumulator = 0
# comp = (df['epoch{}'.format(EPOCH)] == new_df['epoch{}'.format(EPOCH)])
# new_df.to_csv("loaded.csv")