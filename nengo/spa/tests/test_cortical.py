import pytest
import nengo
import nengo.spa as spa
import numpy as np


def test_connect():
    class SPA(spa.SPA):
        def __init__(self):
            spa.SPA.__init__(self)
            self.buffer1 = spa.Buffer(dimensions=16)
            self.buffer2 = spa.Buffer(dimensions=16)
            self.cortical = spa.Cortical(spa.Actions('buffer2=buffer1'))
            self.input = spa.Input(buffer1='A')

    model = SPA()

    output, vocab = model.get_module_output('buffer2')

    with model:
        p = nengo.Probe(output, 'output', filter=0.03)

    sim = nengo.Simulator(model)
    sim.run(0.2)

    match = np.dot(sim.data[p], vocab.parse('A').v)
    assert match[199] > 0.9


def test_transform():
    class SPA(spa.SPA):
        def __init__(self):
            spa.SPA.__init__(self)
            self.buffer1 = spa.Buffer(dimensions=16)
            self.buffer2 = spa.Buffer(dimensions=16)
            self.cortical = spa.Cortical(spa.Actions('buffer2=buffer1*B'))
            self.input = spa.Input(buffer1='A')

    model = SPA()

    output, vocab = model.get_module_output('buffer2')

    with model:
        p = nengo.Probe(output, 'output', filter=0.03)

    sim = nengo.Simulator(model)
    sim.run(0.2)

    match = np.dot(sim.data[p], vocab.parse('A*B').v)
    assert match[199] > 0.7


def test_translate():
    class SPA(spa.SPA):
        def __init__(self):
            spa.SPA.__init__(self)
            self.buffer1 = spa.Buffer(dimensions=16)
            self.buffer2 = spa.Buffer(dimensions=32)
            self.input = spa.Input(buffer1='A')
            self.cortical = spa.Cortical(spa.Actions('buffer2=buffer1'))

    model = SPA()

    output, vocab = model.get_module_output('buffer2')

    with model:
        p = nengo.Probe(output, 'output', filter=0.03)

    sim = nengo.Simulator(model)
    sim.run(0.2)

    match = np.dot(sim.data[p], vocab.parse('A').v)
    assert match[199] > 0.7


def test_errors():
    class SPA(spa.SPA):
        def __init__(self):
            spa.SPA.__init__(self)
            self.buffer = spa.Buffer(dimensions=16)
            self.cortical = spa.Cortical(spa.Actions(
                'dot(buffer,A) --> buffer=buffer'))

    with pytest.raises(NotImplementedError):
        SPA()

    class SPA(spa.SPA):
        def __init__(self):
            spa.SPA.__init__(self)
            self.buffer = spa.Buffer(dimensions=16)
            self.cortical = spa.Cortical(spa.Actions('buffer2=buffer'))

    with pytest.raises(KeyError):
        SPA()


def test_direct():
    class SPA(spa.SPA):
        def __init__(self):
            spa.SPA.__init__(self)
            self.buffer1 = spa.Buffer(dimensions=16)
            self.buffer2 = spa.Buffer(dimensions=32)
            self.cortical = spa.Cortical(spa.Actions(
                'buffer1=A', 'buffer2=B',
                'buffer1=C, buffer2=C'))

    model = SPA()

    output1, vocab1 = model.get_module_output('buffer1')
    output2, vocab2 = model.get_module_output('buffer2')

    with model:
        p1 = nengo.Probe(output1, 'output', filter=0.03)
        p2 = nengo.Probe(output2, 'output', filter=0.03)

    sim = nengo.Simulator(model)
    sim.run(0.2)

    match1 = np.dot(sim.data[p1], vocab1.parse('A+C').v)
    match2 = np.dot(sim.data[p2], vocab2.parse('A+C').v)
    assert match1[199] > 0.3
    assert match2[199] > 0.3

if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
