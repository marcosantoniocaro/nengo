import numpy as np
import pytest

import nengo
import nengo.spa as spa

def test_spa_basic():

    class SpaTestBasic(spa.SPA):
        class Rules:
            def a():
                match(state='A')
                effect(state='B', state2=state*10)
            def b():
                match(state='B')
                effect(state='C', state2=state*10)
            def c():
                match(state='C')
                effect(state='D', state2=state*10)
            def d():
                match(state='D')
                effect(state='E', state2=state*10)
            def e():
                match(state='E')
                effect(state='A', state2=state*10)

        def __init__(self):
            spa.SPA.__init__(self)
            self.state = spa.Memory(dimensions=32)
            self.state2 = spa.Memory(dimensions=32)

            self.bg = spa.BasalGanglia(rules=self.Rules)
            self.thal = spa.Thalamus(self.bg)

            def state_input(t):
                if t<0.1: return 'A'
                else: return '0'
            self.input = spa.Input(self.state, state_input)

    model = nengo.Network()
    with model:
        s = SpaTestBasic(label='spa')
        print s._modules

        pState = nengo.Probe(s.state.state.output, 'output', filter=0.03)
        pState2 = nengo.Probe(s.state2.state.output, 'output', filter=0.03)
        pRules = nengo.Probe(s.thal.rules.output, 'output', filter=0.03)

    sim = nengo.Simulator(model)
    sim.run(1)

    vectors = s.get_module_output('state')[1].vectors.T
    import pylab
    pylab.subplot(3, 1, 1)
    pylab.plot(np.dot(sim.data[pState], vectors))
    pylab.subplot(3, 1, 2)
    pylab.plot(np.dot(sim.data[pState2], vectors))
    pylab.subplot(3, 1, 3)
    pylab.plot(sim.data[pRules])
    pylab.show()



if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
