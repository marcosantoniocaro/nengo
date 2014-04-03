import nengo

class Module(nengo.Network):
    def __init__(self, *args, **kwargs):
        self.inputs = {}
        self.outputs = {}
        nengo.Network.__init__(self, *args, **kwargs)

    def on_add(self, spa):
        pass
