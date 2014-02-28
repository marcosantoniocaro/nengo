import nengo
from .. import objects
from .base import Module

class Memory(Module):
    def make(self, dimensions, subdimensions=16, neurons_per_dimension=50, 
                        filter=0.01, vocab=None):
    
        if vocab is None:
            vocab = dimensions 
            
        self.state = nengo.networks.EnsembleArray(
                                nengo.LIF(neurons_per_dimension*subdimensions),
                                dimensions/subdimensions,
                                dimensions=subdimensions, label='state')
                               
        nengo.Connection(self.state.output, self.state.input, filter=filter)
        
        self.inputs = dict(default=(self.state.input, vocab))
        self.outputs = dict(default=(self.state.output, vocab))
        
        

