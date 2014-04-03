import nengo
from .. import objects
from .module import Module

import numpy as np

class Compare(Module):
    def make(self, dimensions,
                        vocab=None, neurons_per_multiply=200,
                        output_scaling=1.0, radius=1.0):

        if vocab is None:
            vocab = dimensions

        self.output_scaling = output_scaling

        self.compare = nengo.networks.EnsembleArray(
            nengo.LIF(neurons_per_multiply), length=dimensions, dimensions=2,
            label='compare')

        encoders = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype='float')/np.sqrt(2)
        encoders = np.tile(encoders, ((neurons_per_multiply/4)+1,1))[:neurons_per_multiply]
        for e in self.compare.ensembles:
            e.encoders = encoders
            e.radius = radius*np.sqrt(2)

        self.inputA = nengo.Node(size_in=dimensions, label='inputA')
        self.inputB = nengo.Node(size_in=dimensions, label='inputB')

        self.output = nengo.Node(size_in=dimensions, label='output')


        self.inputs = dict(A=(self.inputA, vocab), B=(self.inputB, vocab))
        self.outputs = dict(default=(self.output, vocab))

        t1=np.zeros((dimensions*2,dimensions),dtype='float')
        t2=np.zeros((dimensions*2,dimensions),dtype='float')
        for i in range(dimensions):
            t1[i*2,i]=1.0
            t2[i*2+1,i]=1.0

        nengo.Connection(self.inputA, self.compare.input, transform=t1)
        nengo.Connection(self.inputB, self.compare.input, transform=t2)

        def multiply(x):
            return [x[0]*x[1]]
        self.compare.add_output('product', 1, function=multiply)



    def on_add(self, spa):
        Module.on_add(self, spa)

        vocab = self.outputs['default'][1]

        transform = np.array([vocab.parse('YES').v for i in range(vocab.dimensions)])

        nengo.Connection(self.compare.product, self.output,
                transform=transform.T*self.output_scaling)
