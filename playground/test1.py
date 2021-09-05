import torch as t
import time

time1 = time.time()
a = t.rand(100)
b = t.zeros(100)
indices = a.topk(20).indices
perm = t.randperm(20)
idx = perm[:10]
indices = indices[idx]
b[indices] = a[indices]
time2 = time.time()
print(b)
print(time2 - time1)
