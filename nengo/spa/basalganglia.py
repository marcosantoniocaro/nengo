import nengo
from .module import Module
from .action_condition import DotProduct, Source
import numpy as np


class BasalGanglia(nengo.networks.BasalGanglia, Module):
    """A Basal Ganglia, performing action selection on a set of given actions.

    Parameters
    ----------
    actions : spa.Actions
        The actions to choose between
    input_filter : float
        The filter on all input connections
    """
    def __init__(self, actions, input_filter=0.002):
        self.actions = actions
        self.input_filter = input_filter
        self._bias = None
        Module.__init__(self)
        nengo.networks.BasalGanglia.__init__(self,
                                             dimensions=self.actions.count)

    @property
    def bias(self):
        """Create a bias node, when needed."""
        if self._bias is None:
            with self:
                self._bias = nengo.Node([1])
        return self._bias

    def on_add(self, spa):
        """Form the connections into the BG to compute the utilty values.

        Each action's condition variable contains the set of computations
        needed for that action's utility value, which is the input to the
        basal ganglia.
        """
        Module.on_add(self, spa)
        self.spa = spa

        self.actions.process(spa)   # parse the actions

        for i, action in enumerate(self.actions.actions):
            cond = action.condition.condition
            # the basal ganglia hangles the condition part of the action;
            # the effect is handled by the thalamus

            # Note: A Source is an output from a module, and a Symbol is
            # text that can be parsed to be a SemanticPointer

            for c in cond.items:
                if isinstance(c, DotProduct):
                    if isinstance(c.item1, Source):
                        if isinstance(c.item2, Source):
                            # dot product between two different sources
                            self.add_compare_input(i, c.item1, c.item2,
                                                   c.scale)
                        else:
                            self.add_dot_input(i, c.item1, c.item2, c.scale)
                    else:
                        assert isinstance(c.item2, Source)
                        self.add_dot_input(i, c.item2, c.item1, c.scale)
                else:
                    # Just a fixed scalar input
                    assert isinstance(c, (int, float))
                    self.add_bias_input(i, c)

    def add_bias_input(self, index, value):
        """Make an input that is just a fixed scalar value.

        Parameters
        ----------
        index : int
            the index of the action
        value : float, int
            the fixed utility value to add
        """
        with self.spa:
            nengo.Connection(self.bias, self.input[index:index+1],
                             transform=value, filter=self.input_filter)

    def add_compare_input(self, index, source1, source2, scale):
        """Make an input that is the dot product of two different sources.

        This would be used for an input action such as "dot(vision, memory)"
        Each source might be transformed before being compared.  If the
        two sources have different vocabularies, we use the vocabulary of
        the first one for comparison.

        Parameters
        ----------
        index : int
            the index of the action
        source1 : Source
            the first module output to read from
        source2 : Source
            the second module output to read from
        scale : float
            a scaling factor to be applied to the result
        """
        raise NotImplementedError('Compare connections not implemented yet')

    def add_dot_input(self, index, source, symbol, scale):
        """Make an input that is the dot product of a Source and a Symbol.

        This would be used for an input action such as "dot(vision, A)"
        The source may have a transformation applied first.

        Parameters
        ----------
        index : int
            the index of the action
        source : Source
            the module output to read from
        symbol : Source
            the semantic pointer to compute the dot product with
        scale : float
            a scaling factor to be applied to the result
        """
        output, vocab = self.spa.get_module_output(source.name)
        # the first transformation, to handle dot(vision*A, B)
        t1 = vocab.parse(source.transform.symbol).get_convolution_matrix()
        # the linear transform to compute the fixed dot product
        t2 = np.array([vocab.parse(symbol.symbol).v*scale])

        transform = np.dot(t2, t1)

        with self.spa:
            nengo.Connection(output, self.input[index:index+1],
                             transform=transform, filter=self.input_filter)
