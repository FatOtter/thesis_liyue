from constants import *
from participant import ShallowCNN
import torch
import math
import pandas as pd
from torch.utils.data import DataLoader


class Visualizer:
    """
    Visualizer class to create raw data using for visualization
    """
    def __init__(self, data):
        """
        Initialize the visualizer with data used for visualization
        :param data: The data set used to calculate loss for visualization
        """
        # The temp model used to calculate the loss value
        self.temp_model = ShallowCNN()
        self.optimizer = torch.optim.Adam(self.temp_model.model.parameters())
        self.loss_func = torch.nn.CrossEntropyLoss()

        # The random vectors used to generate plots
        self.random_vec1 = ShallowCNN()
        self.random_vec2 = ShallowCNN()
        self.vec_tensor = torch.empty(1)

        # The data used for plotting
        self.temp_model.set_test_data(data)

        # A python map to store calculated values
        self.loss_map = {}

        # The anchor as the center of the 2d plotting area
        self.anchor = None

        # Resolution
        self.width = 50
        self.height = 50
        self.scale = 1.5

        # Clean up the loss map values
        self.clear_loss_map()

    def clear_loss_map(self):
        """
        Initialize the current loss_map values
        :return: None
        """
        self.loss_map = {'alpha': [], 'beta': [], 'loss': [], 'x': [], 'y': []}

    def Goodfellow_approach(self, theta1=None, theta2=None, scale=1.2, filter_normalization=False, resolution=100,
                            print_progress=True):
        """
        Generate the loss landscape cross-section using Goodfellow's approach, with filter normalization function
        recommended by Tom Goldstein's team
        :param theta1: Given model containing set of parameters as the first point, None to use a random point
        :param theta2: Given model containing set of parameters as the second point, None to use a random point
        :param scale: The factor describes how much each side of the cross section will scale up, must >= 1
        :param filter_normalization: True to apply filter normalization, False not
        :param resolution: the number of sampling points between theta1 and theta2
        :param print_progress: True to print the current visualization progress
        :return: Pandas DataFrame object with alpha factor and corresponding loss values
        """
        # Initialize class variables
        if theta1 is not None:
            if isinstance(theta1, ShallowCNN):
                self.random_vec1 = theta1
            else:
                raise TypeError("Theta1 is not a valid instance of model")
        else:
            self.random_vec1.random_init()
        if theta2 is not None:
            if isinstance(theta2, ShallowCNN):
                self.random_vec2 = theta2
            else:
                raise TypeError("Theta2 is not a valid instance of model")
        else:
            self.random_vec2.random_init()
        self.clear_loss_map()

        # Set up bounds for iterator
        low_bound = math.floor(resolution * (1 - scale))
        up_bound = math.floor(resolution * scale)
        print_step = resolution//10

        # Iterate to sampling over the two points
        for i in range(low_bound, up_bound):
            alpha = i / resolution
            vec1 = self.random_vec1.model.parameters()
            vec2 = self.random_vec2.model.parameters()
            for param in self.temp_model.model.parameters():
                target = alpha * next(vec1) + (1 - alpha) * next(vec2)
                if filter_normalization:
                    target = target * torch.linalg.norm(param) / torch.linalg.norm(target)
                with torch.no_grad():
                    param.copy_(target)
            loss = self.temp_model.get_test_outcome()
            self.loss_map['alpha'].append(alpha)
            self.loss_map['loss'].append(loss)
            if print_progress and i % print_step == 0:
                print(
                    "Overall {} samples, currently {} has complete. Current alpha {}, current loss {}.".format(
                        math.floor(resolution * (2 * scale - 1)),
                        i - low_bound,
                        alpha,
                        loss
                    ))
        result = pd.DataFrame()
        result['alpha'] = self.loss_map['alpha']
        result['loss'] = self.loss_map['loss']
        return result

    def loss_landscape(self, anchor=None, theta1=None, theta2=None, width=None, height=None, scale=None,
                       filter_normalization=True, print_progress=True, anchor_difference=True,
                       record_parameters=False, direction_vec_normalization=False):
        """
        Generate the data for loss landscape over the two directions defined by theta1 and theta2. If not given then use
        random initialization.
        :param anchor: The model with parameters as the center point of the landscape
        :param theta1: The model with parameters as the first direction of axis, randomly initialized if None is given
        :param theta2: The model with parameters as the second direction of axis, randomly initialized if None is given
        :param width: The width of this landscape
        :param height: The height of this landscape
        :param scale: The scale of this landscape, indicating how far landscape will go over the given directions
        :param filter_normalization: True to apply filter normalization, False will not
        :param print_progress: True to print the current progress, False will not
        :param anchor_difference: True if subtract the parameter value from anchor when calculating current point
        :param record_parameters: True to generate a csv file containing parameters each sampled point
        :param direction_vec_normalization: True to apply normalization for direction vectors
        :return: pandas.DataFrame object containing the loss landscape data
        """

        # Initialize class variables
        if theta1 is not None:
            if isinstance(theta1, ShallowCNN):
                self.random_vec1 = theta1
            else:
                raise TypeError("Theta1 is not a valid instance of model")
        if theta2 is not None:
            if isinstance(theta2, ShallowCNN):
                self.random_vec2 = theta2
            else:
                raise TypeError("Theta2 is not a valid instance of model")
        if self.vec_tensor.size()[0] != self.random_vec2.get_flatten_parameter().size()[0]:
            self.vec_tensor = torch.cat((self.random_vec1.get_flatten_parameter(),
                                         self.random_vec2.get_flatten_parameter()))
            self.vec_tensor = self.vec_tensor.reshape(self.random_vec2.get_flatten_parameter().size()[0], -1)
        print("The direction vector size: {}".format(self.vec_tensor.size()))
        if anchor is not None:
            self.set_anchor(anchor)
        if self.anchor is None:
            raise TypeError("Anchor not set")
        self.clear_loss_map()
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height
        if scale is not None:
            self.scale = scale
        parameter_data = {}

        # Iterate to sampling space
        for alpha in range(self.width):
            for beta in range(self.height):
                alpha_factor = (alpha - self.width/2)/(self.width/self.scale)
                beta_factor = (beta - self.height/2)/(self.height/self.scale)
                param1 = self.random_vec1.model.parameters()
                param2 = self.random_vec2.model.parameters()
                anchor_params = self.anchor.model.parameters()

                # Call the helper function to load parameters
                self._load_parameters_to_temp(alpha_factor, beta_factor, anchor_params, param1, param2,
                                              anchor_difference, filter_normalization, direction_vec_normalization)
                loss = self.temp_model.get_test_outcome()

                # Get the axis according to matrix multiplication
                diff_vec = self.temp_model.get_flatten_parameter() - self.anchor.get_flatten_parameter()
                axis = torch.matmul(diff_vec.detach(), self.vec_tensor.type(torch.float))
                x = axis[0].item()
                y = axis[1].item()

                # Record the values
                self.loss_map['alpha'].append(alpha_factor)
                self.loss_map['beta'].append(beta_factor)
                self.loss_map['loss'].append(loss)
                self.loss_map['x'].append(x)
                self.loss_map['y'].append(y)
                if print_progress:
                    print("Alpha: {}, Beta:{}, X={}, Y={}, Loss:{} ...".format(alpha, beta, x, y, loss))

                # Save the parameters if necessary
                if record_parameters:
                    parameter_data[(alpha, beta)] = self.temp_model.get_flatten_parameter().detach().numpy()
        if record_parameters:
            parameter_df = pd.DataFrame(parameter_data)
            parameter_df.to_csv(RECORDING_PATH+"PCA_parameters_"+time_str+".csv")
        return pd.DataFrame(self.loss_map)

    def _load_parameters_to_temp(self, alpha_factor, beta_factor, anchor_params, param1, param2,
                                 anchor_difference, filter_normalization, direction_vector_normalization):
        """
        Load alpha and beta factors, calculate the current theta under given alpha and beta, to generate a temp model
        with given alpha and beta.
        Instruction of parameters in above loss_landscape function
        """
        for param in self.temp_model.model.parameters():
            anchor_param = next(anchor_params)
            # Check if make difference with anchor
            if anchor_difference:
                vec1 = next(param1) - anchor_param
                vec2 = next(param2) - anchor_param
            else:
                vec1 = next(param1)
                vec2 = next(param2)
            if not filter_normalization:
                # The version without filter normalization
                temp_param = anchor_param + alpha_factor * vec1 \
                             + beta_factor * vec2
            else:
                # Add filter normalization before visualization
                if direction_vector_normalization:
                    vec1 = vec1 * torch.linalg.norm(anchor_param) / torch.linalg.norm(vec1)
                    vec2 = vec2 * torch.linalg.norm(anchor_param) / torch.linalg.norm(vec2)
                temp_param = anchor_param + alpha_factor * vec1 + beta_factor * vec2
                temp_param = temp_param * torch.linalg.norm(anchor_param) / torch.linalg.norm(temp_param)
            with torch.no_grad():
                param.copy_(temp_param)

    def set_anchor(self, anchor: ShallowCNN):
        """
        Set the anchor for this visualizer
        :param anchor: center point for loss landscape
        :return: None
        """
        self.anchor = anchor

    def get_axis(self, alpha_factor, beta_factor):
        """
        The reverse calculation from alpha and beta factor to the axis coordinator
        :param alpha_factor: the projection value of the given vector on visualization direction
        :param beta_factor: the projection value of the given vector on visualization direction
        :return: the axis of trajectory to draw on the landscape diagram
        """
        alpha = (alpha_factor/self.scale)*(self.width/2) + self.width/2
        beta = (beta_factor/self.scale)*(self.height/2) + self.height/2
        return round(alpha), round(beta)

    def get_projection(self, target: ShallowCNN):
        """
        Show the projection of the target point on the loss landscape
        :param target: The target point to calculate the projection
        :return: tuple<int, int> show the coordinators of the given target on the loss landscape
        """
        direction1 = self.get_flatten_vec(self.random_vec1)
        direction2 = self.get_flatten_vec(self.random_vec2)
        target_vec = self.get_flatten_vec(target)
        alpha_factor = torch.matmul(target_vec, direction1) / torch.norm(direction1)
        beta_factor = torch.matmul(target_vec, direction2) / torch.norm(direction2)
        return self.get_axis(alpha_factor.item(), beta_factor.item())

    def get_flatten_vec(self, module: ShallowCNN, normalize=False, minus_anchor=True):
        """
        Get the flattened vector of the given module parameters as a tensor
        :param minus_anchor: True to minus anchor before return, false will return the flatten vector directly
        :param module: The module to get flatten vector
        :param normalize: Indicate if apply normalization according to anchor or not
        :return: flattened vector of the given module parameters as a tensor
        """
        direction = torch.empty(1)
        anchor = self.anchor.parameters()
        with torch.no_grad():
            for param in module.model.parameters():
                anchor_param = next(anchor)
                if minus_anchor:
                    param = param - anchor_param
                param = param.flatten()
                if normalize:
                    param = param * torch.linalg.norm(anchor_param) / torch.linalg.norm(param)
                direction = torch.cat([direction, param])
        return direction[1:]

    def get_directions(self):
        """
        Generate a DataFrame with parameters of random directions in it
        :return: DataFrame with parameters of random directions in it
        """
        result = pd.DataFrame()
        result['vec1'] = self.get_flatten_vec(self.random_vec1, False, False).numpy()
        result['vec2'] = self.get_flatten_vec(self.random_vec2, False, False).numpy()
        return result

    def get_distance_to_anchor(self, model: ShallowCNN):
        """
        Return the distance from given model to the anchor
        :param model: The model to calculate distance
        :return: Distance to anchor
        """
        vec = self.get_flatten_vec(model)
        return torch.linalg.norm(vec)

    def load_random_vectors(self, data: pd.DataFrame):
        """
        load vectors from saved csv file
        :param data: data frame store saved vectors
        :return: None
        """
        self.random_vec1.load_parameters(data, "vec1")
        self.random_vec1.load_parameters(data, "vec2")

    def get_loss_at_axis(self, alpha: int, beta: int, filter_normalization=True, anchor_difference=True):
        """
        Get the loss value (and accuracy) at a given point
        :param alpha: the alpha axis on the plot
        :param beta: the beta axis on the plot
        :param filter_normalization: True to apply filter normalization
        :return: tuple<float, float> indicating the (loss_value, accuracy) at a given point
        """
        alpha_factor = (alpha - self.width / 2) / (self.width / self.scale)
        beta_factor = (beta - self.height / 2) / (self.height / self.scale)
        print("Alpha factor = {}, Beta factor = {}".format(alpha_factor, beta_factor))
        param1 = self.random_vec1.model.parameters()
        param2 = self.random_vec2.model.parameters()
        anchor_params = self.anchor.parameters()
        self._load_parameters_to_temp(alpha_factor, beta_factor, anchor_params, param1, param2,
                                      anchor_difference, filter_normalization)
        loss, acc = self.temp_model.get_test_outcome(True)
        distance = self.get_distance_to_anchor(self.temp_model)
        norm = self.temp_model.get_parameter_norm()
        return loss, acc, distance, norm

    def init_pca(self, df: pd.DataFrame, x_start=0, y_start=1, anchor_idx=-1, save_coords=True):
        """
        Initialize the direction with PCA applied to the input data frame as a trajectory file
        """
        # Load the input data frame as a tensor
        trajectory = torch.tensor(df.to_numpy()[x_start:, y_start:])
        trajectory = trajectory.transpose(0, 1)
        anchor = torch.clone(trajectory[anchor_idx])

        # Load anchor
        anchor_df = pd.DataFrame(anchor.numpy())
        self.anchor = ShallowCNN()
        self.anchor.load_parameters(anchor_df, 0)
        anchor_diff = self.anchor.get_flatten_parameter() - anchor
        print("Anchor loaded with difference {} to the original".format(anchor_diff.norm()))

        # Make difference of the trajectory with the defined anchor (usually final epoch parameters)
        trajectory -= anchor
        trajectory = trajectory[:anchor_idx]
        trajectory = trajectory.transpose(0, 1)
        print(trajectory.size())

        # Conduct PCA process
        u, s, v = torch.pca_lowrank(trajectory)
        self.vec_tensor = u[:, :2]
        print("The calculated vector size: {}".format(self.vec_tensor.size()))
        selected_directions = pd.DataFrame(self.vec_tensor.numpy())
        self.random_vec1.load_parameters(selected_directions, 0)
        self.random_vec2.load_parameters(selected_directions, 1)
        loaded_diff1 = self.vec_tensor[:, 0] - self.random_vec1.get_flatten_parameter()
        loaded_diff2 = self.vec_tensor[:, 1] - self.random_vec2.get_flatten_parameter()
        print("Loaded parameters difference with original tensor: {}, {}".format(loaded_diff1.norm(), loaded_diff2.norm()))

        # Save the trajectory to a csv file if necessary
        if save_coords:
            coords = torch.matmul(trajectory.transpose(0,1), self.vec_tensor)
            acc_recorder = torch.zeros(0)
            for i in range(len(trajectory[0])):
                print("\rOverall {} trajectory points, now calculating {}".format(len(trajectory[0]), i), end="")
                self.temp_model.load_parameters(trajectory[:, i])
                loss, acc = self.temp_model.get_test_outcome(True)
                acc_recorder = torch.cat([acc_recorder, torch.tensor((loss, acc)).unsqueeze(0)])
            coords = torch.cat([coords, acc_recorder], dim=1)
            print("Trajectory file preview: \n", coords)
            pd.DataFrame(coords.detach().numpy(), columns=["x", "y", "loss", "acc"]).to_csv(RECORDING_PATH+"Trajectory"+time_str+".csv")
