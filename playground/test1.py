import numpy as np

PRE_TRAINED_CIFAR10_PATH = "../datasets/pretrained_cifar10/cifar100_resnet20_"
indices = np.random.permutation(19)[:3]
paths = [PRE_TRAINED_CIFAR10_PATH+"test{}.csv".format(x) for x in indices]
print(paths)
samples = np.vstack(
    [np.genfromtxt(x, delimiter=',') for x in paths]
)
print(samples.shape)

