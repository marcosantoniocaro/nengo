import pytest
import nengo
import nengo.spa as spa
import numpy as np


def test_thalamus():
    class SPA(spa.SPA):
        def __init__(self):
            spa.SPA.__init__(self, rng=np.random.RandomState(2))
            self.vision = spa.Buffer(dimensions=16)
            self.vision2 = spa.Buffer(dimensions=16)
            self.motor = spa.Buffer(dimensions=16)
            self.motor2 = spa.Buffer(dimensions=32)

            actions = spa.Actions(
                'dot(vision, A) --> motor=A, motor2=vision*vision2',
                'dot(vision, B) --> motor=vision, motor2=vision*A*~B',
                )
            self.bg = spa.BasalGanglia(actions)
            self.thalamus = spa.Thalamus(self.bg)

            def input(t):
                if t < 0.1:
                    return 'A'
                elif t < 0.3:
                    return 'B'
                else:
                    return '0'
            self.input = spa.Input(vision=input, vision2='B*~A')

    model = SPA()

    with model:
        input, vocab = model.get_module_input('motor')
        input2, vocab2 = model.get_module_input('motor2')
        p = nengo.Probe(input, 'output', filter=0.03)
        p2 = nengo.Probe(input2, 'output', filter=0.03)

    model.seed = 3
    sim = nengo.Simulator(model)
    sim.run(0.3)

    data = vocab.dot(sim.data[p].T)
    data2 = vocab2.dot(sim.data[p2].T)

    assert 0.9 < data[0, 100] < 1.1
    assert -0.2 < data[0, 299] < 0.2
    assert 0.9 < data[1, 299] < 1.1
    assert -0.2 < data[1, 100] < 0.2
    assert 0.6 < data2[0, 299] < 0.8
    assert -0.2 < data2[0, 100] < 0.2
    assert -0.2 < data2[1, 299] < 0.2
    assert 0.4 < data2[1, 100] < 0.6


def test_errors():
    class SPA(spa.SPA):
        def __init__(self):
            spa.SPA.__init__(self, rng=np.random.RandomState(2))
            self.vision = spa.Buffer(dimensions=16)

            actions = spa.Actions(
                '0.5 --> motor=A'
                )
            self.bg = spa.BasalGanglia(actions)
    with pytest.raises(NameError):
        SPA()


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
