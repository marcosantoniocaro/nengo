import nengo
import nengo.spa


class Module(nengo.Network):
    """Base class for SPA Modules.

    Modules are Networks that also have a list of inputs and outputs,
    each with an associated Vocabulary (or a desired dimensionality for
    the Vocabulary).

    The inputs and outputs are dictionaries that map a name to an
    (object, Vocabulary) pair.  The object can be a Node or an Ensemble.
    """

    def __init__(self):
        self.inputs = {}
        self.outputs = {}

    def on_add(self, spa):
        """Called when this is assigned to a variable in the SPA network.

        Overload this when you want processing to be delayed until after
        the Module is attached to the SPA network.
        """
        pass
