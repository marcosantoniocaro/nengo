import pytest

import nengo
import nengo.spa as spa
import numpy as np

from nengo.tests.helpers import Plotter


def test_spa_complex():
    model = nengo.Network()

    dimensions = 64

    class ParseWrite(spa.SPA):
        actions = spa.Actions(
            'dot(vision, WRITE) --> verb=vision, verb_latch=1',
            'dot(vision, ONE+TWO+THREE) --> noun=vision, noun_latch=1',
            '''0.5*(dot(NONE-WRITE-ONE-TWO-THREE, vision) +
                    dot(phrase, WRITE*VERB))
               --> motor=phrase*~NOUN''',
            )

        cortical_actions = spa.Actions(
            'phrase=noun*NOUN + verb*VERB',
            )

        def __init__(self):
            spa.SPA.__init__(self)
            self.vision = spa.Buffer(dimensions=dimensions)
            self.phrase = spa.Buffer(dimensions=dimensions)
            self.motor = spa.Buffer(dimensions=dimensions)

            self.noun = spa.DoubleLatch(dimensions=dimensions)
            self.verb = spa.DoubleLatch(dimensions=dimensions)

            self.bg = spa.BasalGanglia(actions=self.actions)
            self.thal = spa.Thalamus(self.bg)

            def input_vision(t):
                index = int(t/0.5)
                sequence = 'WRITE ONE NONE WRITE TWO NONE THREE WRITE NONE'.split()
                if index >= len(sequence):
                    index = len(sequence)-1
                return sequence[index]
            self.input = spa.Input(vision=input_vision)

            self.cortical = spa.Cortical(self.cortical_actions)

    with model:
        s = ParseWrite(label='SPA')

        probes = {
            #'vision': nengo.Probe(s.vision.state.output, filter=0.03),
            'vision': nengo.Probe(s.noun.input, filter=0.03),
            'phrase': nengo.Probe(s.phrase.state.output, filter=0.03),
            'motor': nengo.Probe(s.motor.state.output, filter=0.03),
            'noun': nengo.Probe(s.noun.state.output, filter=0.03),
            'verb': nengo.Probe(s.verb.state.output, filter=0.03),
        }
        latches = [nengo.Probe(s.noun.latch, filter=0.03),
                   nengo.Probe(s.verb.latch, filter=0.03)]
        products = [nengo.Probe(s.noun.latched, filter=0.03),
                   nengo.Probe(s.verb.latched, filter=0.03)]
    sim = nengo.Simulator(model)
    sim.run(4.5)

    import pylab as plt
    for i, module in enumerate('vision noun verb phrase motor'.split()):
        plt.subplot(6, 1, i+1)
        plt.plot(np.dot(sim.data[probes[module]],
                        s.get_module_output(module)[1].vectors.T))
        plt.legend(s.get_module_output(module)[1].keys, fontsize='xx-small')
        plt.ylabel(module)

    plt.subplot(6, 1, 6)
    plt.plot(sim.trange(), sim.data[latches[0]], label='Noun Latch')
    plt.plot(sim.trange(), sim.data[latches[1]], label='Verb Latch')
    plt.plot(sim.trange(), sim.data[products[0]], label='Noun Set')
    plt.plot(sim.trange(), sim.data[products[1]], label='Verb Set')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
