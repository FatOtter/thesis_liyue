import torch
from participant import ShallowCNN
from constants import *
import pandas as pd

df = pd.read_csv("./records/Confined_parameters2021_09_03_00.csv")
t = df.to_numpy()
t = t[:, 1:]
# print(t)
t = torch.tensor(t)
t = t.transpose(0, 1)
anchor = t[320]
M = t - anchor
print(M)
print(M.size())
M = M[:320]
M = M.transpose(0, 1)
print(M.size())

u, s, v = torch.pca_lowrank(M)
# vec = u[:, :2].numpy()
# vec_df = pd.DataFrame(vec)
# vec_df.to_csv("pca_diff.csv")

vec = u[:, :2]
trajectory = torch.matmul(vec.transpose(0,1), M)
trajectory = trajectory.transpose(0,1)
print(trajectory)
df = pd.DataFrame(trajectory.numpy())
df.to_csv("./records/trajectory_test"+time_str+".csv")


df = pd.read_csv("records/PCA_parameters_2021_09_04_14.csv")
p = df.to_numpy()
p = p[1:, 1:]
p = torch.tensor(p)
p = p.transpose(0,1)
p -= anchor
print(p.size())
p = p.transpose(0,1)


coords = torch.matmul(vec.transpose(0,1), p)
print(coords)
for i in range(len(p)-1):
    x = torch.matmul(p[:, i], vec)
    print(x)
df = pd.DataFrame(coords.transpose(0,1).numpy())
df.to_csv("./records/trajectory_coords"+time_str+".csv")