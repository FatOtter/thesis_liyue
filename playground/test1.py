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
path = "./records/Confined_acc2021_09_17_16.csv"
raw = np.genfromtxt(path, delimiter=",")
raw = raw[1:]
# print(raw)
x1 = raw[::3, 4]
x2 = raw[1::3, 4]
x3 = raw[2::3, 4]
l1 = raw[::3, 3]
l2 = raw[1::3, 3]
l3 = raw[2::3, 3]
l1 = l1.clip(0,20)
l2 = l2.clip(0,20)
l3 = l3.clip(0,20)
fig, axs = plot.subplots(2)
axs[0].plot(x1, color="red")
axs[0].plot(x2, color="blue")
axs[0].plot(x3, color="green")
axs[1].plot(l1, color="red")
axs[1].plot(l2, color="blue")
axs[1].plot(l2, color="green")
plot.show()
