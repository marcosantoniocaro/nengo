import nengo
from .module import Module
from .action_objects import Symbol, Source

import numpy as np


class Cortical(Module):
    """A SPA module for forming connections between other modules.

    Parameters
    ----------
    actions : spa.Actions
        The actions to implement
    filter : float
        The synaptic filter to use for the connections
    """
    def __init__(self, actions, filter=0.01):
        Module.__init__(self)
        self.actions = actions
        self.filter = filter
        self._bias = None

    def on_add(self, spa):
        Module.on_add(self, spa)
        self.spa = spa

        # parse the provided class and match it up with the spa model
        self.actions.process(spa)
        for action in self.actions.actions:
            if action.condition is not None:
                raise NotImplementedError('Cannot handle conditions on ' +
                                          'cortical actions yet.')
            for name, effects in action.effect.effect.items():
                for effect in effects.items:
                    if isinstance(effect, Symbol):
                        self.add_direct_effect(name, effect.symbol)
                    elif isinstance(effect, Source):
                        self.add_route_effect(name, effect.name,
                                              effect.transform.symbol)
                    else:
                        raise NotImplementedError('Unknown effect %s' %
                                                  effect)

    @property
    def bias(self):
        if self._bias is None:
            with self:
                self._bias = nengo.Node([1])
        return self._bias

    def add_direct_effect(self, target_name, text):
        sink, vocab = self.spa.get_module_input(target_name)
        transform = np.array([vocab.parse(text).v]).T

        with self.spa:
            nengo.Connection(self.bias, sink, transform=transform,
                             filter=self.filter)

    def add_route_effect(self, target_name, source_name, transform):
        target, target_vocab = self.spa.get_module_input(target_name)
        source, source_vocab = self.spa.get_module_output(source_name)

        t = source_vocab.parse(transform).get_convolution_matrix()
        if target_vocab is not source_vocab:
            t = np.dot(source_vocab.transform_to(target_vocab), t)

        with self.spa:
            nengo.Connection(source, target, transform=t,
                             filter=self.filter)
