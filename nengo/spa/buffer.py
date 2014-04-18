import nengo
from .module import Module


class Buffer(Module):
    """A module capable of representing a single vector, with no memory.

    This is a minimal SPA module, useful for passing data along (for example,
    visual input).

    Parameters
    ----------
    dimensions : int
        Number of dimensions for the vector
    subdimensions : int
        Size of the individual ensembles making up the vector.  Must divide
        evenly into dimensions
    neurons_per_dimensions : int
        Number of neurons in an ensemble will be this*subdimensions
    vocab : Vocabulary, optional
        The vocabulary to use to interpret this vector
    direct : boolean
        Whether or not to use direct mode for the neurons
    """
    def __init__(self, dimensions, subdimensions=16, neurons_per_dimension=50,
                 vocab=None, direct=False):
        Module.__init__(self)

        if vocab is None:
            # use the default one for this dimensionality
            vocab = dimensions

        if dimensions % subdimensions != 0:
            raise Exception('Number of dimensions(%d) ' % dimensions +
                            'must be divisible by subdimensions(%d)' %
                            subdimensions)

        if direct:
            neurons = nengo.Direct()
        else:
            neurons = nengo.LIF(neurons_per_dimension*subdimensions)
        self.state = nengo.networks.EnsembleArray(
            neurons,
            dimensions/subdimensions,
            dimensions=subdimensions, label='state')

        self.inputs = dict(default=(self.state.input, vocab))
        self.outputs = dict(default=(self.state.output, vocab))
