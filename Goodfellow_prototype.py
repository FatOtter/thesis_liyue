from datetime import datetime

import torch
import torchvision
import pandas as pd
import math
from torch.utils.data.dataloader import DataLoader
from model_shallow_cnn import ShallowCNN
from read_model_from_csv_test import ModelLoader

RECORDING_PATH = "./trainig_records/"
EPOCH = 4

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.5,],  # mean=[0.5071, 0.4865, 0.4409] for cifar100
        std=[0.5,])
])

train_data = torchvision.datasets.MNIST("./mnist", True, transform, download=True)
test_data = torchvision.datasets.MNIST("./mnist", False, transform)

# model1 = ShallowCNN()
# model2 = ShallowCNN()


class Visualizer:
    def __init__(self, data):
        # The temp model used to calculate the loss value
        self.temp_model = ShallowCNN()
        self.optimizer = torch.optim.Adam(self.temp_model.parameters())
        self.loss_func = torch.nn.CrossEntropyLoss()

        # The random vectors used to generate plots
        self.random_vec1 = ShallowCNN()
        self.random_vec2 = ShallowCNN()

        # The data used for plotting
        self.plot_loader = DataLoader(data, batch_size=64)
        self.data_length = len(data)

        # A python map to store calculated values
        self.loss_map = {}

        # The anchor as the center of the 2d plotting area
        self.anchor = None

        # Clean up the loss map values
        self.clear_loss_map()

    def clear_loss_map(self):
        self.loss_map = {'alpha': [], 'beta': [], 'loss': []}

    # Generate loss value diagram using Goodfellow's approach
    def Goodfellow_visualize(self, model1: torch.nn.Module, model2: torch.nn.Module):
        if model1 is None:
            model1 = self.random_vec1
        if model2 is None:
            model2 = self.random_vec2
        self.clear_loss_map()
        for alpha in range(-20, 120):
            print("Calculating for alpha = {} ....".format(alpha))
            factor = alpha/100
            param1 = model1.parameters()
            param2 = model2.parameters()
            for param in self.temp_model.parameters():
                temp = factor*next(param1)+(1-factor)*(next(param2))
                with torch.no_grad():
                    param.copy_(temp)
            loss = 0
            for batch_x, batch_y in self.plot_loader:
                with torch.no_grad():
                    out = self.temp_model(batch_x)
                loss += self.loss_func(out, batch_y)
            loss = loss/self.data_length
            self.loss_map['alpha'].append(alpha)
            self.loss_map['loss'].append(loss.item())
        return self.loss_map.copy()

    # Set a anchor (center point) for the landscape
    def set_anchor(self, path, column):
        loader = ModelLoader()
        self.anchor = loader.load(path, column)

    # Generate loss value landscape according to the given resolution and scale
    def loss_landscape(self, width: int, height: int, scale=1.2):
        if self.anchor is None:
            raise ValueError("Anchor not set")
        self.clear_loss_map()
        for alpha in range(width):
            for beta in range(height):
                alpha_factor = (alpha - width/2)/(width/scale)
                beta_factor = (beta - height/2)/(height/scale)
                param1 = self.random_vec1.parameters()
                param2 = self.random_vec2.parameters()
                anchor_params = self.anchor.parameters()
                for param in self.temp_model.parameters():
                    anchor_param = next(anchor_params)

                    # Deprecated: the version without filter normalization
                    # temp_param = anchor_param + alpha_factor * (next(param1) - anchor_param) + beta_factor * (next(param2) - anchor_param)

                    # Add filter normalization before visualization
                    vec1 = next(param1) - anchor_param
                    vec1 = vec1 * torch.linalg.norm(anchor_param) / torch.linalg.norm(vec1)
                    vec2 = next(param2) - anchor_param
                    vec2 = vec2 * torch.linalg.norm(anchor_param) / torch.linalg.norm(vec2)
                    temp_param = anchor_param + alpha_factor * vec1 + beta_factor * vec2
                    with torch.no_grad():
                        param.copy_(temp_param)
                loss = 0
                for batch_x, batch_y in self.plot_loader:
                    with torch.no_grad():
                        out = self.temp_model(batch_x)
                    loss += self.loss_func(out, batch_y)
                loss = loss/self.data_length
                self.loss_map['alpha'].append(alpha)
                self.loss_map['beta'].append(beta)
                self.loss_map['loss'].append(loss.item())
                print("Alpha: {}, Beta:{}, Loss:{} ...".format(alpha, beta, loss))
        return self.loss_map.copy()

    def get_projection(self, target: torch.nn.Module):
        direction1 = self.get_flatten_vec(self.random_vec1)
        direction2 = self.get_flatten_vec(self.random_vec2)
        target_vec = self.get_flatten_vec(target)
        alpha_factor = torch.matmul(target_vec, direction1) / torch.norm(direction1)
        beta_factor = torch.matmul(target_vec, direction2) / torch.norm(direction2)
        return alpha_factor.item(), beta_factor.item()

    def get_flatten_vec(self, module: torch.nn.Module):
        direction = torch.empty(1)
        anchor = self.anchor.parameters()
        with torch.no_grad():
            for param in module.parameters():
                anchor_param = next(anchor)
                param = param - anchor_param
                # Feature Normalization (Deprecated)
                # param = param * torch.linalg.norm(anchor_param) / torch.linalg.norm(param)
                param = param.flatten()
                direction = torch.cat([direction, param])
        return direction[1:]

    def get_axis(self, alpha_factor, beta_factor, width, height, scale = 1.2):
        alpha = (alpha_factor/scale)*(width/2) + width/2
        beta = (beta_factor/scale)*(height/2) + height/2
        return math.floor(alpha), math.floor(beta)



path = RECORDING_PATH+"Params2021_07_31_14.csv"
column = 'epoch{}'.format(EPOCH)

visual1 = Visualizer(test_data)
# outcome = visual1.Goodfellow_visualize(model1, model2)

visual1.set_anchor(path, column)
for i in range(5):
    model = ModelLoader().load(path, 'epoch{}'.format(i))
    alpha_factor, beta_factor = visual1.get_projection(model)
    alpha, beta = visual1.get_axis(alpha_factor, beta_factor, 50, 50, 2.5)
    print("Factors: ({}, {}); Axis: ({}, {})".format(alpha_factor, beta_factor, alpha, beta))


# outcome = visual1.loss_landscape(50, 50, 2.5)
# print(outcome)
# df = pd.DataFrame(outcome)
#
# now = datetime.now()
# time_str = now.strftime("%Y_%m_%d_%H")
# df.to_csv(RECORDING_PATH+time_str+"landscape.csv")