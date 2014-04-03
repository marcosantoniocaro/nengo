from . import vocab
from .module import Module
import nengo

class SPA(nengo.Network):
    def __init__(self, *args, **kwargs):
        self._modules = {}
        self._default_vocabs = {}
        nengo.Network.__init__(self, *args, **kwargs)

    def __setattr__(self, key, value):
        nengo.Network.__setattr__(self, key, value)
        if isinstance(value, Module):
            value.label = key
            self._modules[value.label] = value

            for k, (obj, v) in value.inputs.iteritems():
                if type(v)==int:
                    value.inputs[k] = (obj, self.get_default_vocab(v))
            for k, (obj, v) in value.outputs.iteritems():
                if type(v)==int:
                    value.outputs[k] = (obj, self.get_default_vocab(v))
            value.on_add(self)

    def get_default_vocab(self, dimensions):
        if dimensions not in self._default_vocabs:
            self._default_vocabs[dimensions] = vocab.Vocabulary(dimensions)
        return self._default_vocabs[dimensions]

    def get_module_input(self, name):
        if name in self._modules:
            return self._modules[name].inputs['default']
        elif '_' in name:
            module, name = name.rsplit('_', 1)
            return self._modules[module].inputs[name]

    def get_module_output(self, name):
        if name in self._modules:
            return self._modules[name].outputs['default']
        elif '_' in name:
            module, name = name.rsplit('_', 1)
            return self._modules[module].outputs[name]

