import numpy as np
import pytest

import nengo
import nengo.spa as spa

def test_spa_basic():

    class SpaTestBasic(spa.SPA):
        class Rules:
            def a():
                match(state='A')
                effect(state='B')
            def b():
                match(state='B')
                effect(state='C')
            def c():
                match(state='C')
                effect(state='D')
            def d():
                match(state='D')
                effect(state='E')
            def e():
                match(state='E')
                effect(state='A')

        def make(self):
            self.state = spa.Memory(dimensions=32)

            self.bg = spa.BasalGanglia(rules=self.Rules)
            self.thal = spa.Thalamus(self.bg)

            def state_input(t):
                if t<0.1: return 'A'
                else: return '0'
            self.input = spa.Input(self.state, state_input)

    model = nengo.Network()
    with model:
        s = SpaTestBasic(label='spa')

        pState = nengo.Probe(s.state.state.output, 'output', filter=0.03)
        pRules = nengo.Probe(s.thal.rules.output, 'output', filter=0.03)

    sim = nengo.Simulator(model)
    sim.run(1)

    vectors = s.get_module_output('state')[1].vectors.T
    import pylab
    pylab.subplot(2, 1, 1)
    pylab.plot(np.dot(sim.data[pState], vectors))
    pylab.subplot(2, 1, 2)
    pylab.plot(sim.data[pRules])
    pylab.show()



if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
