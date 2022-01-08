import numpy as np
batch_size = 20
j = 8

a = np.delete(range(int(batch_size)), j)
print(a)