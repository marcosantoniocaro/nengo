import pytest
import nengo
import nengo.spa as spa
import numpy as np


def test_basic():
    class Basic(spa.SPA):
        def __init__(self):
            spa.SPA.__init__(self)
            self.compare = spa.Compare(dimensions=16)

    model = Basic()

    inputA = model.get_module_input('compare_A')
    inputB = model.get_module_input('compare_B')
    output = model.get_module_output('compare')
    assert inputA[0] is model.compare.inputA
    assert inputB[0] is model.compare.inputB
    assert output[0] is model.compare.output
    assert inputA[1] is inputB[1]
    assert inputA[1] is output[1]
    assert inputA[1].dimensions == 16


def test_run():
    class Basic(spa.SPA):
        def __init__(self):
            spa.SPA.__init__(self)
            self.compare = spa.Compare(dimensions=16)

            def inputA(t):
                if 0 <= t < 0.1:
                    return 'A'
                else:
                    return 'B'

            self.input = spa.Input(compare_A=inputA, compare_B='A')
    model = Basic()

    compare, vocab = model.get_module_output('compare')

    with model:
        p = nengo.Probe(compare, 'output', synapse=0.03)

    sim = nengo.Simulator(model, seed=4)
    sim.run(0.2)

    data = np.dot(sim.data[p], vocab.parse('YES').v)

    assert data[100] > 0.8
    assert data[199] < 0.2


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
