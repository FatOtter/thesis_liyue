import torch
import torchvision
from __init__ import *


class DataDistributor:
    """
    Distribute defined data set (vertically) according to given number of participants
    """
    def __init__(self, number_of_participants=10, data_set="MNIST", data_set_path=DATA_SET_PATH):
        """
        Initialize the data distributor according to given parameters
        :param number_of_participants: The overall participants to distribute data to
        :param data_set: The data set used for distribution
        :param data_set_path: The file directory to store data sets
        """
        self.number_of_participants = number_of_participants
        self.train_set = None
        self.test_set = None
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.5, ],  # mean=[0.5071, 0.4865, 0.4409] for cifar100
                std=[0.5, ])
        ])
        if data_set == "MNIST":
            self.train_set = torchvision.datasets.MNIST(data_set_path, True, transform, download=True)
            self.test_set = torchvision.datasets.MNIST(data_set_path, False, transform)
        else:
            raise NotImplementedError("Data set is not implemented yet")
        self.split_train = None
        self.split_test = None
        self.split_data()

    def split_data(self):
        """
        Distribute data according to the given number of participants, split data stored in self.split_train and
        self.split_test
        :return: None
        """
        train_samples_per_node = len(self.train_set) // self.number_of_participants
        test_samples_per_node = len(self.test_set) // self.number_of_participants
        train_samples_count = []
        test_samples_count = []
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