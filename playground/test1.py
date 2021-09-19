import numpy as np

sample = np.vstack(np.genfromtxt("../datasets/pretrained_cifar10/cifar100_resnet20_train.csv", delimiter=','))
print(sample.shape)


