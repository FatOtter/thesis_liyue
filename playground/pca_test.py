import torch
from participant import ShallowCNN
import pandas as pd

df = pd.read_csv("../anchor.csv")
t = df.to_numpy()
t = t[1:, 1:]
# print(t)
t = torch.tensor(t)
t = t.transpose(0, 1)
anchor = t[-1]
M = t - anchor
print(M)
print(M.size())
M = M[:-1]
M = M.transpose(0, 1)
print(M.size())

u, s, v = torch.pca_lowrank(M)
# vec = u[:, :2].numpy()
# vec_df = pd.DataFrame(vec)
# vec_df.to_csv("pca_diff.csv")

vec = u[:, :2]

df = pd.read_csv("./records/PCA_parameters_2021_08_27_00.csv")
p = df.to_numpy()
p = p[1:, 1:]
p = torch.tensor(p)
p = p.transpose(0,1)
p -= anchor
print(p.size())
p = p.transpose(0,1)


coords = torch.matmul(vec.transpose(0,1), p)
print(coords)
df = pd.DataFrame(coords.transpose(0,1).numpy())
df.to_csv("pca_coords_3.csv")