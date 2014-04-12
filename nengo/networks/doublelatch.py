import numpy as np

import nengo
from nengo.utils.distributions import Uniform

class DoubleLatch(nengo.Network):

    def __init__(self, dimensions, subdimensions=16, neurons_per_dimension=100,
                 speed=0.2, pstc_feedback=0.005, pstc_hard=0.02,
                 pstc_inhibit=0.001, neurons_inhibit=50, inhibit_amount=20,
                 latch_threshold=0.90, disable_latch_threshold=0.95,
                 disable_latch_gain=2, **kwargs):
        super(DoubleLatch, self).__init__(**kwargs)

        if dimensions % subdimensions != 0:
            raise ValueError("Number of dimensions(%d) "
                            "must be divisible by subdimensions(%d)" %
                            (dimensions, subdimensions))

        n_neurons = neurons_per_dimension*subdimensions
        n_partitions = dimensions/subdimensions
        radius = kwargs.pop("radius", 1.0)
        array_radius = 1.3*np.sqrt(radius / n_partitions)
        dot_radius = radius**2
        self.dimensions = dimensions

        self.input = nengo.Node(size_in=dimensions, label="input")
        self.latch = nengo.Node(size_in=1, label="latch")
        self.dot = nengo.Node(size_in=1, label="dot")
        self.latched = nengo.Node(size_in=1, label="latched")

        # When latch is > latch_threshold, inhibit does not fire, and so
        # error is uninhibited => state converges to input. Otherwise,
        # inhibit fires, and and state remains the same.
        self.inhibit_high = nengo.Ensemble(
            nengo.LIF(neurons_inhibit), dimensions=1,
            encoders=[[1]]*neurons_inhibit,
            intercepts=Uniform(1 - latch_threshold, 1.0),
            label="inhibit_high")
        self.inhibit_low = nengo.Ensemble(
            nengo.LIF(neurons_inhibit), dimensions=1,
            encoders=[[1]]*neurons_inhibit,
            intercepts=Uniform(latch_threshold, 1.0),
            label="inhibit_low")
        self.hard_latch = nengo.Ensemble(
            nengo.LIF(neurons_inhibit), dimensions=1, radius=dot_radius,
            encoders=[[1]]*neurons_inhibit,
            intercepts=Uniform(disable_latch_threshold, 1.0),
            label="hard_latch")
        self.inhibit_latched = nengo.Ensemble(
            nengo.LIF(neurons_inhibit), dimensions=1,
            encoders=[[1]]*neurons_inhibit,
            intercepts=Uniform(0.5, 1.0),
            label="inhibit_latched")

        self.error = nengo.networks.EnsembleArray(
            nengo.LIF(n_neurons), n_partitions, dimensions=subdimensions,
            radius=array_radius, label="error", **kwargs)
        self.state = nengo.networks.EnsembleArray(
            nengo.LIF(n_neurons), n_partitions, dimensions=subdimensions,
            radius=array_radius, label="state", **kwargs)
        self.product = nengo.networks.Product(
            nengo.LIF(n_neurons), dimensions, radius=radius)

        nengo.Connection(self.input, self.error.input, filter=None)
        nengo.Connection(self.state.output, self.error.input, transform=-1)
        nengo.Connection(self.error.output, self.state.input,
                         transform=speed)
        nengo.Connection(self.state.output, self.state.input,
                         filter=pstc_feedback)

        nengo.Connection(nengo.Node(output=[1]), self.inhibit_high)
        nengo.Connection(self.latch, self.inhibit_high, transform=-1)

        nengo.Connection(self.input, self.product.A)
        nengo.Connection(self.state.output, self.product.B)
        nengo.Connection(self.product.output, self.dot,
                         transform=self.product.dot_product_transform(),
                         filter=None)

        nengo.Connection(nengo.Node(output=[1]), self.inhibit_low)
        nengo.Connection(self.latch, self.inhibit_low, transform=-1)
        nengo.Connection(self.dot, self.hard_latch)
        nengo.Connection(self.hard_latch, self.hard_latch,
                         transform=disable_latch_gain, filter=pstc_hard)
        nengo.Connection(self.inhibit_low.neurons, self.hard_latch.neurons,
                         filter=pstc_inhibit,
                         transform=-inhibit_amount*np.ones(
                            (neurons_inhibit, neurons_inhibit)))
        #nengo.Connection(self.hard_latch, self.latched,
        #                 function=lambda x: 1.0/disable_latch_gain,
        #                 filter=None)

        pstc_latched = 0.005
        nengo.Connection(nengo.Node(output=lambda t: int(t > pstc_latched)),
                         self.latched)
        nengo.Connection(nengo.Node(output=[1]), self.inhibit_latched,
                         filter=None)
        nengo.Connection(self.inhibit_latched, self.latched, transform=-1,
                         filter=pstc_latched)
        nengo.Connection(self.hard_latch.neurons, self.inhibit_latched.neurons,
                         filter=pstc_inhibit,
                         transform=-inhibit_amount*np.ones(
                             (neurons_inhibit, neurons_inhibit)))

        for ensemble in self.error.ensembles:
            nengo.Connection(
                self.inhibit_high.neurons, ensemble.neurons,
                filter=pstc_inhibit,
                transform=-inhibit_amount*np.ones((ensemble.n_neurons,
                                                   neurons_inhibit)))
            nengo.Connection(
                self.hard_latch.neurons, ensemble.neurons,
                filter=pstc_inhibit,
                transform=-inhibit_amount*np.ones((ensemble.n_neurons,
                                                   neurons_inhibit)))
