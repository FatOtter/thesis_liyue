import torch
import pandas as pd
import numpy as np


class ShallowCNN(torch.nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()
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

    def write_parameters(self, data_frame: pd.DataFrame, column):
        all_param = torch.empty(1)
        for param in self.parameters():
            param = param.flatten()
            all_param = torch.cat([all_param, param])
        data_frame[column] = all_param[1:].numpy()
        return data_frame
