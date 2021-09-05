import torch
from participant import ShallowCNN


class Aggregator:
    """
    The aggregator class collecting gradients calculated by participants and plus together
    """
    def __init__(self, sample_gradients: torch.Tensor):
        """
        Initiate the aggregator according to the tensor size of a given sample
        """
        self.sample_gradients = sample_gradients
        self.collected_gradients = torch.zeros(sample_gradients.size())
        self.counter = 0
        self.global_model = ShallowCNN()

    def reset(self):
        """
        Reset the aggregator to 0
        """
        self.collected_gradients = torch.zeros(self.sample_gradients.size())
        self.counter = 0

    def collect(self, gradient: torch.Tensor):
        """
        Collect one set of gradients from a participant
        """
        self.collected_gradients += gradient
        self.counter += 1

    def get_outcome(self, reset=False):
        """
        Get the aggregated gradients and reset the aggregator if needed
        """
        result = self.collected_gradients / self.counter
        if reset:
            self.reset()
        return result

    def get_global_model(self):
        """
        Get the global model
        """
        return self.global_model