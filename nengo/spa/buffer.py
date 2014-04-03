import nengo
from .. import objects
from .module import Module

class Buffer(Module):
    def __init__(self, dimensions, subdimensions=16, neurons_per_dimension=50,
                        vocab=None):
        Module.__init__(self)

        if vocab is None:
            vocab = dimensions

        self.state = nengo.networks.EnsembleArray(
                                nengo.LIF(neurons_per_dimension*subdimensions),
                                dimensions/subdimensions,
                                dimensions=subdimensions, label='state')

        self.inputs = dict(default=(self.state.input, vocab))
        self.outputs = dict(default=(self.state.output, vocab))



