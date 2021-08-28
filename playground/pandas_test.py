import pandas as pd
import numpy as np
from datetime import datetime
import torch

RECORDING_PATH = "../trainig_records/"
# test_frame = pd.DataFrame(columns=['epoch1'])
# # for epoch in range(5):
# #     for participant in range(10):
# #         loss = np.abs(np.random.randn(1)[0])
# #         acc = np.random.randint(0,100,1)[0]/100
# #         test_frame.loc[len(test_frame)] = [epoch, participant, loss, acc]
# existing = torch.empty(1)
# print(existing.size())
# for i in range(5):
#     new = torch.randn((16,8,3,3))
#     new = new.flatten()
#     print(new.size())
#     existing = torch.cat([existing, new])
# test_frame['epoch1'] = existing.numpy()
#
# now = datetime.now()
# time_str = now.strftime("%Y_%m_%d_%H")
# test_frame.to_csv(RECORDING_PATH+time_str+".csv")

# test_frame = pd.read_csv(RECORDING_PATH+"Vectors2021_08_03_19.csv")
# tensor = torch.tensor(test_frame['vec2'].to_numpy())
# print("Average={}, Norm={}".format(tensor.sum()/tensor.size()[0], tensor.norm()))
dict = {}
for i in range(10):
    for j in range(10):
        dict[(i, j)] = torch.rand(10000).numpy()
df = pd.DataFrame(dict)
df.to_csv("pandas_dict_test.csv")