import numpy as np

from nengo.neurons import LIF
from nengo.networks.associative import HeteroAssociative
from nengo.spa.module import Module


class Cleanup(HeteroAssociative, Module):
    """TODO"""

    def __init__(self, vocab, neurons_per_pointer=100, intercept=0.1,
                 use_all_encoders=True, **kwargs):
        keys = vocab.vectors

        super(Cleanup, self).__init__(
            LIF(neurons_per_pointer*len(keys)),
            self.calculate_max_capacity(vocab.dimensions, intercept),
            initial_keys=keys, **kwargs)

        self.inputs = dict(default=(self.key, self.dimension))
        self.outputs = dict(default=(self.output, self.dimension))
