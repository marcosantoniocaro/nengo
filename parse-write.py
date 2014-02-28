import nengo
import nengo.spa as spa

model = nengo.Model('Parse-Write')

dimensions = 64          

class ParseWrite(spa.SPA):
    class Rules:
        def verb():
            match(vision='WRITE')
            effect(verb=vision)
        def noun():
            match(vision='ONE+TWO+THREE')
            effect(noun=vision)        
        def write():
            match(vision='0.5*(NONE-WRITE-ONE-TWO-THREE)', phrase='0.5*WRITE*VERB')
            effect(motor=phrase*'~NOUN')
            
    class CorticalRules:
        def noun():
            effect(phrase=noun*'NOUN')
        def verb():
            effect(phrase=verb*'VERB')
            
    
    def make(self):
        self.vision = spa.Buffer('vision', dimensions=dimensions)
        self.phrase = spa.Buffer('phrase', dimensions=dimensions)
        self.motor = spa.Buffer('motor', dimensions=dimensions)

        self.noun = spa.Memory('noun', dimensions=dimensions)
        self.verb = spa.Memory('verb', dimensions=dimensions)
    
        self.bg = spa.BasalGanglia('bg', rules=self.Rules)
        self.thal = spa.Thalamus('thal', 'bg')
        
        def input_vision(t):
            index = int(t/0.5)
            sequence = 'WRITE ONE NONE WRITE TWO NONE THREE WRITE NONE'.split()
            if index >= len(sequence): 
                index = len(sequence)-1
            return sequence[index]    
        self.input = spa.Input('vision', input_vision)
        
        self.cortical = spa.Cortical('cortical', self.CorticalRules)
        
with model:        
    s = ParseWrite(label='SPA')

probes = {
    'vision': nengo.Probe(s.vision.state.output, filter=0.03),
    'phrase': nengo.Probe(s.phrase.state.output, filter=0.03),
    'motor': nengo.Probe(s.motor.state.output, filter=0.03),
    'noun': nengo.Probe(s.noun.state.output, filter=0.03),
    'verb': nengo.Probe(s.verb.state.output, filter=0.03),
}
sim = nengo.Simulator(model)
sim.run(4.5)
              
import numpy as np
import pylab


for i, module in enumerate('vision noun verb phrase motor'.split()):
    pylab.subplot(5, 1, i+1)
    pylab.plot(np.dot(sim.data(probes[module]), s.get_module_output(module)[1].vectors.T))
    pylab.legend(s.get_module_output(module)[1].keys, fontsize='xx-small')
    pylab.ylabel(module)
pylab.show()
              
