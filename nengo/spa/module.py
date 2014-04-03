import nengo
import nengo.spa


class Module(nengo.Network):
    def __init__(self, *args, **kwargs):
        self.inputs = {}
        self.outputs = {}
        nengo.Network.__init__(self, *args, **kwargs)

    def on_add(self, spa):
        for k, (obj, v) in self.inputs.iteritems():
            if type(v) == int:
                self.inputs[k] = (obj, spa.get_default_vocab(v))
        for k, (obj, v) in self.outputs.iteritems():
            if type(v) == int:
                self.outputs[k] = (obj, spa.get_default_vocab(v))
