import numpy as np

import nengo
from nengo.utils.distributions import Uniform
from nengo.networks.associative import AutoAssociative


class DoubleLatch(nengo.Network):

    def __init__(self, dimensions, neurons_per_dimension=100, speed=1.0,
                 pstc_state=0.05, pstc_hard=0.02, pstc_latched=0.005,
                 pstc_inhibit=0.001, neurons_inhibit=100, inhibit_amount=20,
                 latch_threshold=0.95, double_latch_threshold=0.95,
                 double_latch_gain=3, **kwargs):
        super(DoubleLatch, self).__init__(**kwargs)

        radius = kwargs.pop("radius", 1.0)
        if dimensions == 1:
            array_radius = radius
        else:
            # make each input dimension have a 99% chance of being in
            # array_radius
            array_radius = radius * AutoAssociative._calculate_intercept(
                dimensions, 0.01)
        dot_radius = radius**2
        self.dimensions = dimensions

        self.input = nengo.Node(size_in=dimensions, label="input")
        self.latch = nengo.Node(size_in=1, label="latch")
        self.latched = nengo.Node(size_in=1, label="latched")
        self.error = nengo.Node(size_in=dimensions, label="error")
        self.dot = nengo.Ensemble(nengo.LIF(neurons_per_dimension), 1, encoders=[[1]]*neurons_per_dimension, intercepts=Uniform(0, 1), label="dot") #Node(size_in=1, label="dot")

        # When latch is > latch_threshold, inhibit does not fire, and so
        # error is uninhibited => state converges to input. Otherwise,
        # inhibit fires, and and state remains the same.
        self.inhibit_high = nengo.Ensemble(
            nengo.LIF(neurons_inhibit), dimensions=1,
            encoders=[[1]]*neurons_inhibit,
            intercepts=Uniform(1 - latch_threshold, 1),
            label="inhibit_high")
        self.inhibit_low = nengo.Ensemble(
            nengo.LIF(neurons_inhibit), dimensions=1,
            encoders=[[1]]*neurons_inhibit,
            intercepts=Uniform(latch_threshold, 1),
            label="inhibit_low")
        self.hard_latch = nengo.Ensemble(
            nengo.LIF(neurons_inhibit), dimensions=1, radius=dot_radius,
            encoders=[[1]]*neurons_inhibit,
            intercepts=Uniform(double_latch_threshold, 1),
            label="hard_latch")
        self.inhibit_latched = nengo.Ensemble(
            nengo.LIF(neurons_inhibit), dimensions=1,
            encoders=[[1]]*neurons_inhibit,
            intercepts=Uniform(0.5, 1),
            label="inhibit_latched")

        self.inhibited_error = nengo.networks.EnsembleArray(
            nengo.LIF(neurons_per_dimension), n_ensembles=dimensions,
            dimensions=1, radius=array_radius, label="error",
            intercepts=Uniform(0, 1), **kwargs)
        self.state = nengo.networks.EnsembleArray(
            nengo.LIF(neurons_per_dimension), n_ensembles=dimensions,
            dimensions=1, radius=array_radius, label="state", **kwargs)
        self.product = nengo.networks.Product(
            nengo.LIF(2*neurons_per_dimension), dimensions, radius=radius)

        nengo.Connection(self.input, self.error, filter=None)
        nengo.Connection(self.state.output, self.error, transform=-1,
                         filter=pstc_state)
        nengo.Connection(self.error, self.inhibited_error.input, filter=None)
        nengo.Connection(self.inhibited_error.output, self.state.input,
                         transform=speed)
        nengo.Connection(self.state.output, self.state.input,
                         filter=pstc_state)

        nengo.Connection(nengo.Node(output=[1]), self.inhibit_high)
        nengo.Connection(self.latch, self.inhibit_high, transform=-1)

        nengo.Connection(self.error, self.product.A, filter=None)
        nengo.Connection(self.error, self.product.B, filter=None)
        nengo.Connection(self.product.output, self.dot, filter=0.008,
                         transform=self.product.dot_product_transform(-1))

        nengo.Connection(nengo.Node(output=[1]), self.inhibit_low)
        nengo.Connection(self.latch, self.inhibit_low, transform=-1)
        nengo.Connection(nengo.Node(output=[1]), self.dot, filter=pstc_hard)
        nengo.Connection(self.dot, self.hard_latch, filter=pstc_hard)
        nengo.Connection(self.hard_latch, self.hard_latch,
                         transform=double_latch_gain, filter=pstc_hard,
                         eval_points=[[1]])
        nengo.Connection(self.inhibit_low.neurons, self.hard_latch.neurons,
                         filter=pstc_inhibit,
                         transform=-inhibit_amount*np.ones(
                             (neurons_inhibit, neurons_inhibit)))
        nengo.Connection(self.inhibit_high, self.dot, transform=-inhibit_amount)

        nengo.Connection(nengo.Node(output=[1]), self.latched,
                         filter=pstc_latched)
        nengo.Connection(nengo.Node(output=[1]), self.inhibit_latched,
                         filter=None)
        nengo.Connection(self.inhibit_latched, self.latched, transform=-1,
                         filter=pstc_latched)
        nengo.Connection(self.hard_latch.neurons, self.inhibit_latched.neurons,
                         filter=pstc_inhibit,
                         transform=-inhibit_amount*np.ones(
                             (neurons_inhibit, neurons_inhibit)))

        for ensemble in self.inhibited_error.ensembles:
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
