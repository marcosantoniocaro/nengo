import nengo
from .module import Module
from .action_condition import DotProduct, Source


class BasalGanglia(nengo.networks.BasalGanglia, Module):
    def __init__(self, actions, input_filter=0.002):
        self.actions = actions
        self.input_filter = input_filter
        self.bias = None
        Module.__init__(self)

        nengo.networks.BasalGanglia.__init__(self,
                                             dimensions=self.actions.count)

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
        raise NotImplementedError('Compare connections not implemented yet')

    def add_dot_input(self, index, source, symbol, scale):
        source, vocab = self.spa.get_module_output(source.name)
        transform = [vocab.parse(symbol.symbol).v*scale]

        with self.spa:
            nengo.Connection(source, self.input[index:index+1],
                             transform=transform, filter=self.input_filter)
