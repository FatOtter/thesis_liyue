import torch as t
import torch.autograd as autograd

a = t.rand(2, 10000)
b = t.rand(2, 10000)
x = t.cat((a, b))
print(x.size())
