from . import vocab
from ..objects import Network

class Module(Network):
    def __init__(self, *args, **kwargs):
        self.inputs = {}
        self.outputs = {}
        Network.__init__(self, *args, **kwargs)

    def on_add(self, spa):
        for k, (obj, v) in self.inputs.iteritems():
            if type(v)==int:
                self.inputs[k] = (obj, spa.get_default_vocab(v))
        for k, (obj, v) in self.outputs.iteritems():
            if type(v)==int:
                self.outputs[k] = (obj, spa.get_default_vocab(v))


class SPA(Network):
    def __init__(self, *args, **kwargs):
        self.modules = {}
        self.default_vocabs = {}
        Network.__init__(self, *args, **kwargs)

    def __setattr__(self, key, value):
        Network.__setattr__(self, key, value)
        if isinstance(value, Module):
            value.label = key
            self.modules[value.label] = value
            value.on_add(self)

    def get_default_vocab(self, dimensions):
        if dimensions not in self.default_vocabs:
            self.default_vocabs[dimensions] = vocab.Vocabulary(dimensions)
        return self.default_vocabs[dimensions]

    def get_module_input(self, name):
        if name in self.modules:
            return self.modules[name].inputs['default']
        elif '_' in name:
            module, name = name.split('_', 1)
            return self.modules[module].inputs[name]

    def get_module_output(self, name):
        if name in self.modules:
            return self.modules[name].outputs['default']
        elif '_' in name:
            module, name = name.split('_', 1)
            return self.modules[module].outputs[name]

