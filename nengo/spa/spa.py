from . import vocab
from .module import Module
import nengo


class SPA(nengo.Network):
    """Base class for SPA models.

    This expands the standard Network system to support structured connections
    that use Semantic Pointers and associated vocabularies in their
    definitions.

    To build a SPA model, subclass this SPA class and in the make() method
    add in your objects.  Any spa.Module object that is assigned to a
    member variable will automatically be accessible by the SPA connection
    system.  For example, the following code will build three modules
    (two Buffers and a Memory) that can be referred to as a, b, and c,
    respectively:

    class Example(spa.SPA):
        def make(self):
            self.a = spa.Buffer(dimensions=8)
            self.b = spa.Buffer(dimensions=16)
            self.c = spa.Memory(dimensions=8)

    These names can be used by special Modules that are aware of these
    names.  For example, the Cortical module allows you to form connections
    between these modules in ways that are aware of semantic pointers:

    class Example(spa.SPA):
        class CorticalRules:
            def rule1():
                effect(b=a*'CAT')
            def rule2():
                effect(c=b*'~CAT')

        def make(self):
            self.a = spa.Buffer(dimensions=8)
            self.b = spa.Buffer(dimensions=16)
            self.c = spa.Memory(dimensions=8)
            self.cortical = spa.Cortical(CorticalRules)
    """

    def __init__(self, *args, **kwargs):
        # the set of known modules
        self._modules = {}
        # the Vocabulary to use by default for a given dimensionality
        self._default_vocabs = {}
        nengo.Network.__init__(self, *args, **kwargs)

    def __setattr__(self, key, value):
        """A setattr that handles Modules being added specially.

        This is so that we can use the variable name for the Module as
        the name that all of the SPA system will use to access that module.
        """
        nengo.Network.__setattr__(self, key, value)
        if isinstance(value, Module):
            value.label = key
            self._modules[value.label] = value
            value.on_add(self)
            for k, (obj, v) in value.inputs.iteritems():
                if type(v) == int:
                    value.inputs[k] = (obj, self.get_default_vocab(v))
            for k, (obj, v) in value.outputs.iteritems():
                if type(v) == int:
                    value.outputs[k] = (obj, self.get_default_vocab(v))

    def get_default_vocab(self, dimensions):
        """Return a Vocabulary with the desired dimensions.

        This will create a new default Vocabulary if one doesn't exist.
        """
        if dimensions not in self._default_vocabs:
            self._default_vocabs[dimensions] = vocab.Vocabulary(dimensions)
        return self._default_vocabs[dimensions]

    def get_module_input(self, name):
        """Return the object to connect into for the given name.

        The name will be either the same as a module, or of the form
        <module_name>_<input_name>.
        """
        if name in self._modules:
            return self._modules[name].inputs['default']
        elif '_' in name:
            module, name = name.rsplit('_', 1)
            return self._modules[module].inputs[name]

    def get_module_output(self, name):
        """Return the object to connect into for the given name.

        The name will be either the same as a module, or of the form
        <module_name>_<output_name>.
        """
        if name in self._modules:
            return self._modules[name].outputs['default']
        elif '_' in name:
            module, name = name.rsplit('_', 1)
            return self._modules[module].outputs[name]
