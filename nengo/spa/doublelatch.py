import numpy as np

import nengo
from nengo.spa.module import Module

class DoubleLatch(nengo.networks.DoubleLatch, Module):
    """TODO
    """
    def __init__(self, *args, **kwargs):
        vocab = kwargs.pop('vocab', None)
        super(DoubleLatch, self).__init__(*args, **kwargs)

        if vocab is None:
            vocab = self.dimensions

        self.inputs = dict(default=(self.input, vocab), latch=(self.latch, 1))

        self.outputs = dict(default=(self.state.output, vocab),
                            latched=(self.latched, 1),
                            dot=(self.dot, 1))
