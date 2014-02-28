import nengo
from .. import objects
from .base import Module

class Input(Module):
    def __init__(self, target_name, value):
        kwargs = dict(target_name=target_name, value=value)
        Module.__init__(self, 'input_%s'%target_name, **kwargs)
        
    def make(self, target_name, value):
        self.target_name = target_name
        self.value = value
    
    def on_add(self, spa):
        Module.on_add(self, spa)
        
        target, vocab = spa.get_module_input(self.target_name)
        if callable(self.value):
            val = lambda t: vocab.parse(self.value(t)).v
            self.input = nengo.Node(val, label='input')
        else:
            val = vocab.parse(self.value).v
            self.input = nengo.Node(val, label='input')
    
        nengo.Connection(self.input, target, filter=None)        
        

