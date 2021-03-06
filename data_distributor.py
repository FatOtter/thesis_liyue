import torch
import torchvision
import numpy as np
from constants import *

PRE_TRAINED_CIFAR10_PATH = "./datasets/pretrained_cifar10/cifar100_resnet20_"
TOTAL_PRE_TRAINED_SETS = 19
SETS_ACQUIRED = 3

class DataDistributor:
    """
    Distribute defined data set (vertically) according to given number of participants
    """
    def __init__(self, number_of_participants=PARTICIPANTS, data_set=DEFAULT_DATA_SET, data_set_path=DATA_SET_PATH,
                 balanced=True):
        """
        Initialize the data distributor according to given parameters
        :param number_of_participants: The overall participants to distribute data to
        :param data_set: The data set used for distribution
        :param data_set_path: The file directory to store data sets
        """
        self.number_of_participants = number_of_participants
        self.train_set = None
        self.test_set = None
        MNIST_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.5, ],  # mean=[0.5071, 0.4865, 0.4409] for cifar100
                std=[0.5, ])
        ])
        CIFAR_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, ]
            )
        ])
        if data_set == "MNIST":
            self.train_set = torchvision.datasets.MNIST(data_set_path, True, MNIST_transform, download=True)
            self.test_set = torchvision.datasets.MNIST(data_set_path, False, MNIST_transform)
        elif data_set == "CIFAR-10":
            self.train_set = torchvision.datasets.CIFAR10(data_set_path, True, CIFAR_transform, download=True)
            self.test_set = torchvision.datasets.CIFAR10(data_set_path, False, MNIST_transform)
        elif data_set == "Pre-trained CIFAR-10":
            self.load_pre_trained_cifar10()
        else:
            raise NotImplementedError("Data set is not implemented yet")
        self.split_train = None
        self.split_test = None
        self.split_data(balanced)
        print("Data loaded to distributor, {} train samples, {} test samples"
              .format(len(self.train_set), len(self.test_set)))

    def split_data(self, balanced=True):
        """
        Distribute data according to the given number of participants, split data stored in self.split_train and
        self.split_test
        :return: None
        """
        train_samples_per_node = len(self.train_set) // self.number_of_participants
        test_samples_per_node = len(self.test_set) // self.number_of_participants
        train_samples_count = []
        test_samples_count = []
        if balanced:
            for i in range(self.number_of_participants):
                if i < self.number_of_participants - 1:
                    train_samples_count.append(train_samples_per_node)
                    test_samples_count.append(test_samples_per_node)
                else:
                    train_samples_count.append(
                        len(self.train_set) - (self.number_of_participants - 1) * train_samples_per_node
                    )
                    test_samples_count.append(
                        len(self.test_set) - (self.number_of_participants - 1) * test_samples_per_node
                    )
        else:
            rands = torch.randint(20, 100, (self.number_of_participants, ))
            for i in range(self.number_of_participants):
                train_samples_count.append(round((len(self.train_set) * rands[i] / sum(rands)).item()))
                test_samples_count.append(round((len(self.test_set) * rands[i] / sum(rands)).item()))
            train_diff = len(self.train_set) - sum(train_samples_count)
            test_diff = len(self.test_set) - sum(test_samples_count)
            train_samples_count[-1] += train_diff
            test_samples_count[-1] += test_diff

        self.split_train = torch.utils.data.random_split(self.train_set, train_samples_count)
        self.split_test = torch.utils.data.random_split(self.test_set, test_samples_count)

    def get_train_data(self, participant_number):
        """
        Get the train data according to the given participant number
        :param participant_number: The participant number to retrieve data
        :return: allocated data set for given participant
        """
        if participant_number < 0 or participant_number >= self.number_of_participants:
            raise ValueError("Invalid participant number")
        return self.split_train[participant_number]

    def get_test_data(self, participant_number):
        """
        Get the train data according to the given participant number
        :param participant_number: The participant number to retrieve data
        :return: allocated data set for given participant
        """
        if participant_number < 0 or participant_number >= self.number_of_participants:
            raise ValueError("Invalid participant number")
        return self.split_test[participant_number]

    def load_pre_trained_cifar10(self):
        """
        Load pre-trained CIFAR10 according to the MNIST like format
        """
        sample_file_index = np.random.permutation(TOTAL_PRE_TRAINED_SETS)[:SETS_ACQUIRED]
        test_paths = [PRE_TRAINED_CIFAR10_PATH + "test{}.csv".format(x) for x in sample_file_index]
        train_paths = [PRE_TRAINED_CIFAR10_PATH + "train{}.csv".format(x) for x in sample_file_index]
        test_samples = torch.tensor(np.vstack(
            [np.genfromtxt(x, delimiter=',') for x in test_paths]
        ))
        train_samples = torch.tensor(np.vstack(
            [np.genfromtxt(x, delimiter=',') for x in train_paths]
        ))
        test_features = test_samples[:, :-1]
        test_features = test_features.float()
        test_label = test_samples[:, -1]
        test_label = test_label.long()
        train_features = train_samples[:, :-1]
        train_features = train_features.float()
        train_label = train_samples[:, -1]
        train_label = train_label.long()
        self.test_set = [(test_features[i], test_label[i]) for i in range(len(test_label))]
        self.train_set = [(train_features[i], train_label[i]) for i in range(len(train_label))]
