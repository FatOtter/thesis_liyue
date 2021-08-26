import torch
import pandas as pd
from torch.utils.data import DataLoader
from constants import *


class ShallowCNN(torch.nn.Module):
    """
    The module used for verification, currently only support MNIST data set
    """
    def __init__(self):
        super(ShallowCNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,8,3,1,1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(8,16, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(16*3*3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())
        self.test_data = None
        self.test_data_length = 0
        self.train_data = None
        self.train_data_length = 0

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out

    def write_parameters(self, data_frame: pd.DataFrame, column):
        """
        Write the parameters of the current model into the defined column of given DataFrame
        :param data_frame: The DataFrame to write data into
        :param column: The column name to write data into
        :return: A DataFrame with parameters in the given column
        """
        all_param = torch.empty(1)
        for param in self.parameters():
            param = param.flatten()
            all_param = torch.cat([all_param, param])
        data_frame[column] = all_param[1:].detach().numpy()
        return data_frame

    def load_parameters(self, data_frame: pd.DataFrame, column, start_index=0):
        """
        Load parameters from the given column of the DataFrame
        :param data_frame: The DataFrame to retrieve parameters
        :param column: The column to load parameters f
        :param start_index: The index of the column to start loading parameters
        :return: None
        """
        for param in self.parameters():
            length = len(param.flatten())
            to_load = data_frame[column][start_index:start_index+length].to_numpy()
            to_load = torch.tensor(to_load)
            to_load = to_load.reshape(param.size())
            with torch.no_grad():
                param.copy_(to_load)
            start_index += length

    def set_training_data(self, training_data, batch_size=DEFAULT_BATCH_SIZE, **kwargs):
        """
        Set the training data for this model according to the given training data
        :param training_data: data set used to train this model
        :param batch_size: the batch size of each training iteration
        :param kwargs: other parameters supported by DataLoader
        :return: None
        """
        self.train_data = DataLoader(training_data, batch_size=batch_size, **kwargs)
        self.train_data_length = len(training_data)

    def set_test_data(self, test_data, batch_size=DEFAULT_BATCH_SIZE, **kwargs):
        """
        Set the training data for this model according to the given training data
        :param test_data: data set used to test this model
        :param batch_size: the batch size of each test iteration
        :param kwargs: other parameters supported by DataLoader
        :return: None
        """
        self.test_data = DataLoader(test_data, batch_size=batch_size, **kwargs)
        self.test_data_length = len(test_data)

    def get_test_outcome(self, calc_acc=False):
        """
        Get the model overall loss value from the test data set
        :param calc_acc Indicate if accuracy need to be calculated, default false to reduce calculation amount
        :return: the outcome value for loss function and the accuracy as tuple<float, float> (loss_value, accuracy)
        """
        if self.test_data is None:
            raise TypeError("Test data not initialized")
        test_loss = 0
        test_acc = 0
        for batch_x, batch_y in self.test_data:
            with torch.no_grad():
                out = self(batch_x)
            batch_loss = self.loss_function(out, batch_y)
            test_loss += batch_loss.item()
            if calc_acc:
                prediction = torch.max(out, 1).indices
                batch_acc = (prediction == batch_y).sum()
                test_acc += batch_acc.item()
        test_acc = test_acc/self.test_data_length
        test_loss = test_loss/self.test_data_length
        if calc_acc:
            return test_loss, test_acc
        else:
            return test_loss

    def normal_epoch(self, print_progress=False):
        """
        Run a training epoch in traditional way
        :param print_progress: Set True to print the training progress, default not printing
        :return: The training loss value and training accuracy as tuple<float, float> (loss_value, accuracy)
        """
        if self.test_data is None:
            raise TypeError("Training data not initialized")
        train_loss = 0
        train_acc = 0
        batch_counter = 0
        overall_batches = len(self.train_data)
        for batch_x, batch_y in self.train_data:
            if print_progress and batch_counter % 100 == 0:
                print("Currently training for batch {}, overall {} batches".format(batch_counter, overall_batches))
            batch_counter += 1
            out = self(batch_x)
            batch_loss = self.loss_function(out, batch_y)
            train_loss += batch_loss.item()
            prediction = torch.max(out, 1).indices
            batch_acc = (prediction == batch_y).sum()
            train_acc += batch_acc.item()
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
        if print_progress:
            print("Epoch complete")
        train_acc = train_acc/self.train_data_length
        train_loss = train_loss/self.train_data_length
        return train_loss, train_acc

    def get_parameter_norm(self):
        """
        Get the norm value for current model
        :return: norm value for flatten current parameter set
        """
        flatten = torch.empty(1)
        for param in self.parameters():
            flatten = torch.cat([flatten, param.flatten()])
        return torch.linalg.norm(flatten[1:])

    def parameter_scale_down(self, scale=0.5):
        """
        Scale down parameters for the current model
        :param scale: the rate to scale
        :return: None
        """
        for param in self.parameters():
            temp = scale * param
            with torch.no_grad():
                param.copy_(temp)

    def confined_init(self, anchor: torch.nn.Module,
                      up_bound=CONFINED_INIT_UP_BOUND, lower_bound=CONFINED_INIT_LOW_BOUND):
        """
        Initialize this model using rules from Confined Gradient Descent
        :param anchor: The center of the initial position
        :param up_bound: The up bound distance
        :param lower_bound: The lower bound distance
        :return: None
        """
        anchors = anchor.parameters()
        delta = torch.rand(1) * (up_bound - lower_bound) + lower_bound
        print("Delta = {}".format(delta.item()))
        for param in self.parameters():
            anchor_vec = next(anchors)
            random_vec = anchor_vec + delta * torch.rand(anchor_vec.size())
            random_vec = random_vec * torch.linalg.norm(anchor_vec) / torch.linalg.norm(random_vec)
            with torch.no_grad():
                param.copy_(random_vec)