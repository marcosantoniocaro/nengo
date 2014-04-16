import numpy as np

import nengo
from nengo.networks import Product
from .module import Module
from .action_condition import DotProduct, Source


class BasalGanglia(nengo.networks.BasalGanglia, Module):
    def __init__(self, actions, input_filter=0.002, n_compare=50, **kwargs):
        self.actions = actions
        self.input_filter = input_filter
        self.bias = None
        self.n_compare = n_compare
        Module.__init__(self)

        nengo.networks.BasalGanglia.__init__(
            self, dimensions=self.actions.count, **kwargs)

    def get_bias_node(self):
        if self.bias is None:
            with self:
                self.bias = nengo.Node([1])
        return self.bias

    def on_add(self, spa):
        Module.on_add(self, spa)
        self.spa = spa

        self.actions.process(spa)

        for i, action in enumerate(self.actions.actions):
            cond = action.condition.condition

            for c in cond.items:
                if isinstance(c, DotProduct):
                    if isinstance(c.item1, Source):
                        if isinstance(c.item2, Source):
                            self.add_compare_input(i, c.item1, c.item2,
                                                   c.scale)
                        else:
                            self.add_dot_input(i, c.item1, c.item2, c.scale)
                    else:
                        assert isinstance(c.item2, Source)
                        self.add_dot_input(i, c.item2, c.item1, c.scale)
                else:
                    assert isinstance(c, (int, float))
                    self.add_bias_input(i, c)

    def add_bias_input(self, index, value):
        with self.spa:
            nengo.Connection(self.get_bias_node(), self.input[index:index+1],
                             transform=value, filter=self.input_filter)

    def add_compare_input(self, index, source1, source2, scale):
        source1, vocab1 = self.spa.get_module_output(source1.name)
        source2, vocab2 = self.spa.get_module_output(source2.name)

        dim = vocab1.dimensions
        if dim != vocab2.dimensions:
            raise ValueError("Cannot compare source1 (%s) of dimension (%d) "
                               "with source2 (%s) of dimension (%d)" % (
                               source1, dim, source2, vocab2.dimensions))

        with self.spa:
            product = Product(nengo.LIF(self.n_compare), dim)
            nengo.Connection(source1, product.A)
            nengo.Connection(source2, product.B)
            nengo.Connection(
                product.output, self.input[index],
                transform=product.dot_product_transform(scale),
                filter=self.input_filter)

    def add_dot_input(self, index, source, symbol, scale):
        source, vocab = self.spa.get_module_output(source.name)
        transform = [vocab.parse(symbol.symbol).v*scale]

        with self.spa:
            nengo.Connection(source, self.input[index],
                             transform=transform, filter=self.input_filter)
