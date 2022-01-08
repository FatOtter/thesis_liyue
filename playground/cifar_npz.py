import numpy as np


# test_raw = np.genfromtxt("../datasets/pretrained_cifar10/cifar100_resnet56_test.csv", delimiter=',')
# print(test_raw[0])
# test_label = np.array(test_raw[:, -1], dtype=np.long)
# test_label_one_hot = np.zeros((test_label.size, test_label.max()+1))
# test_label_one_hot[np.arange(test_label.size), test_label] = 1
# test_data = np.array(test_raw[:, :-1], dtype=np.float)
# train_raw = np.genfromtxt("../datasets/pretrained_cifar10/cifar100_resnet56_train.csv", delimiter=',')
# print(train_raw[0])
# train_label = np.array(train_raw[:, -1], dtype=np.long)
# train_label_one_hot = np.zeros((train_label.size, train_label.max()+1))
# train_label_one_hot[np.arange(train_label.size), train_label] = 1
# train_data = np.array(train_raw[:, :-1], dtype=np.float)
# np.savez_compressed("cifar_resnet56", train_data, train_label, train_label_one_hot, test_data, test_label, test_label_one_hot)


loaded = np.load("cifar_resnet56.npz")
train_data = loaded['arr_0']
train_label = loaded['arr_1']
train_label_one_hot = loaded['arr_2']
print(train_data.shape, train_label.shape, train_label_one_hot.shape)