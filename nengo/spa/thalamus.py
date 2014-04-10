import nengo

from .module import Module
from .action_effect import Symbol, Source

import numpy as np


class Thalamus(Module):
    def __init__(self, bg, neurons_per_rule=50, inhibit=1, pstc_inhibit=0.008,
                 output_filter=0.01, rule_threshold=0.2,
                 neurons_per_channel_dim=50, channel_subdim=16,
                 channel_pstc=0.01, neurons_cconv=200,
                 neurons_gate=40, gate_threshold=0.3,
                 pstc_to_gate=0.002):
        Module.__init__(self)
        self.bg = bg
        self.neurons_per_rule = neurons_per_rule
        self.inhibit = inhibit
        self.pstc_inhibit = pstc_inhibit
        self.output_filter = output_filter
        self.rule_threshold = rule_threshold
        self.neurons_per_channel_dim = neurons_per_channel_dim
        self.channel_subdim = channel_subdim
        self.channel_pstc = channel_pstc
        self.neurons_gate = neurons_gate
        self.neurons_cconv = neurons_cconv
        self.gate_threshold = gate_threshold
        self.pstc_to_gate = pstc_to_gate

        self.gates = {}
        self.channels = {}

    def on_add(self, spa):
        Module.on_add(self, spa)
        self.spa = spa

        N = self.bg.actions.count

        with self:
            actions = nengo.networks.EnsembleArray(
                nengo.LIF(self.neurons_per_rule),
                N, dimensions=1,
                intercepts=nengo.objects.Uniform(self.rule_threshold, 1),
                label='actions')
            self.actions = actions

            for ens in actions.ensembles:
                ens.encoders = [[1.0]] * self.neurons_per_rule

            bias = nengo.Node(output=[1], label='bias')
            self.bias = bias

            nengo.Connection(actions.output, actions.input,
                             transform=(np.eye(N)-1)*self.inhibit,
                             filter=self.pstc_inhibit)

            nengo.Connection(bias, actions.input, transform=np.ones((N, 1)),
                             filter=self.pstc_inhibit)

        with spa:
            nengo.Connection(self.bg.output, actions.input, filter=None)

        for i, action in enumerate(self.bg.actions.actions):
            for name, effects in action.effect.effect.items():
                for effect in effects.items:
                    if isinstance(effect, Symbol):
                        self.add_direct_effect(i, name, effect.symbol)
                    elif isinstance(effect, Source):
                        self.add_route_effect(i, name, effect.name,
                                              effect.transform.symbol)
                    else:
                        raise NotImplementedError('Cannot handle %s' % effect)

    def add_direct_effect(self, index, target_name, text):
        sink, vocab = self.spa.get_module_input(target_name)
        transform = np.array([vocab.parse(text).v]).T

        with self.spa:
            nengo.Connection(self.actions.ensembles[index],
                             sink, transform=transform,
                             filter=self.output_filter)

    def get_gate(self, index):
        if index not in self.gates:
            with self:
                intercepts = nengo.objects.Uniform(self.gate_threshold, 1)
                gate = nengo.Ensemble(nengo.LIF(self.neurons_gate),
                                      dimensions=1,
                                      intercepts=intercepts,
                                      label='gate[%d]' % index,
                                      encoders=[[1]] * self.neurons_gate)
                nengo.Connection(self.actions.ensembles[index], gate,
                                 filter=self.pstc_to_gate, transform=-1)
                nengo.Connection(self.bias, gate, filter=None)
                self.gates[index] = gate

        return self.gates[index]

    def add_route_effect(self, index, target_name, source_name, transform):
        with self:
            gate = self.get_gate(index)

            target, target_vocab = self.spa.get_module_input(target_name)
            source, source_vocab = self.spa.get_module_output(source_name)

            dim = target_vocab.dimensions
            subdim = self.channel_subdim
            assert dim % subdim == 0  # TODO: check this somewhere

            channel = nengo.networks.EnsembleArray(
                nengo.LIF(self.neurons_per_channel_dim*subdim),
                dim/subdim, dimensions=subdim,
                label='channel_%d_%s' % (index, target_name))

            nengo.Connection(channel.output, target, filter=self.channel_pstc)

            inhibit = [[-1]]*(self.neurons_per_channel_dim*subdim)
            for e in channel.ensembles:
                nengo.Connection(gate, e.neurons, transform=inhibit,
                                 filter=self.pstc_inhibit)

            t = source_vocab.parse(transform).get_convolution_matrix()
            if target_vocab is not source_vocab:
                t = np.dot(source_vocab.transform_to(target_vocab), t)

        with self.spa:
            nengo.Connection(source, channel.input, transform=t,
                             filter=self.channel_pstc)

"""



if hasattr(source, 'convolve'):
    # TODO: this is an insanely bizarre computation to have to do
    #   whenever you want to use a CircConv network.  The parameter
    #   should be changed to specify neurons per ensemble
    n_neurons_d = self.neurons_cconv * (
        2*dim - (2 if dim % 2 == 0 else 1))
    channel = nengo.networks.CircularConvolution(
                    nengo.LIF(n_neurons_d), dim,
                    invert_a = source.invert,
                    invert_b = source.convolve.invert,
                    label='cconv_%d_%s'%(index, target.name))

    nengo.Connection(channel.output, target.obj, filter=self.channel_pstc)

    transform = [[-1]]*(self.neurons_cconv)
    for e in channel.ensemble.ensembles:
        nengo.Connection(gate, e.neurons,
                 transform=transform, filter=self.pstc_inhibit)

    # connect first input
    if target.vocab is source.vocab:
        transform = 1
    else:
        transform = source.vocab.transform_to(target.vocab)

    if hasattr(source, 'transform'):
        t2 = source.vocab.parse(source.transform).get_convolution_matrix()
        transform = np.dot(transform, t2)

    nengo.Connection(source.obj, channel.A,
                     transform=transform, filter=self.channel_pstc)

    # connect second input
    if target.vocab is source.convolve.vocab:
        transform = 1
    else:
        transform = source.convolve.vocab.transform_to(target.vocab)

    if hasattr(source.convolve, 'transform'):
        t2 = source.convolve.vocab.parse(source.convolve.transform).
                  get_convolution_matrix()
        transform = np.dot(transform, t2)

    nengo.Connection(source.convolve.obj, channel.B,
                     transform=transform, filter=self.channel_pstc)

                """
