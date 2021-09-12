import torch


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
        self.counter_by_indices = torch.ones(self.sample_gradients.size())

    def reset(self):
        """
        Reset the aggregator to 0
        """
        self.collected_gradients = torch.zeros(self.sample_gradients.size())
        self.counter = 0

    def collect(self, gradient: torch.Tensor, indices=None, sample_count=None):
        """
        Collect one set of gradients from a participant
        """

        if sample_count is None:
            self.collected_gradients += gradient
            if indices is not None:
                self.counter_by_indices[indices] += 1
            self.counter += 1
        else:
            self.collected_gradients += gradient * sample_count
            if indices is not None:
                self.counter_by_indices[indices] += sample_count
            self.counter += sample_count

    def get_outcome(self, reset=False, by_indices=False):
        """
        Get the aggregated gradients and reset the aggregator if needed
        """
        if by_indices:
            result = self.collected_gradients / self.counter_by_indices
        else:
            result = self.collected_gradients / self.counter
        if reset:
            self.reset()
        return result