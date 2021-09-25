import numpy as np
import matplotlib.pyplot as plot

# PRE_TRAINED_CIFAR10_PATH = "../datasets/pretrained_cifar10/cifar100_resnet20_"
# indices = np.random.permutation(19)[:3]
# paths = [PRE_TRAINED_CIFAR10_PATH+"test{}.csv".format(x) for x in indices]
# print(paths)
# samples = np.vstack(
#     [np.genfromtxt(x, delimiter=',') for x in paths]
# )
# print(samples.shape)
path = "./records/Goodfellow2021_09_25_15.csv"
raw = np.genfromtxt(path, delimiter=",")
raw = raw[1:]
print(raw)
plot.plot(raw[:, 1], raw[:, 2])
plot.xlabel("Alpha")
plot.ylabel("loss")
plot.show()