
from .action_condition import Condition
from .action_effect import Effect


class Action(object):
    def __init__(self, sources, sinks, action, name):
        self.name = name
        if '-->' in action:
            condition, effect = action.split('-->', 1)
            self.condition = Condition(sources, condition)
            self.effect = Effect(sources, effect)
        else:
            self.condition = None
            self.effect = Effect(sources, action)
        for name in self.effect.effect.keys():
            if name not in sinks:
                n = self.name
                if n is None:
                    n = action
                raise KeyError('Rule "%s" affects an unknown module "%s"' %
                               (n, name))

    def __str__(self):
        return '<Action %s:\n  Condition: %s\n  Effect: %s\n>' % (self.name,
               self.condition, self.effect)


class Actions(object):
    def __init__(self, *args, **kwargs):
        self.actions = None
        self.args = args
        self.kwargs = kwargs

    @property
    def count(self):
        return len(self.args) + len(self.kwargs)

    def process(self, spa):
        self.actions = []

        sources = list(spa.get_module_outputs())
        sinks = list(spa.get_module_inputs())

        for action in self.args:
            self.actions.append(Action(sources, sinks, action, name=None))
        for name, action in self.kwargs.items():
            self.actions.append(Action(sources, sinks, action, name=name))


if __name__ == '__main__':
    import nengo.spa as spa

    class Test(spa.SPA):
        def __init__(self):
            spa.SPA.__init__(self)
            self.state1 = spa.Buffer(32)
            self.state2 = spa.Buffer(16)

    model = Test()

    actions = Actions(
        'dot(state1, A) --> state1=B',
        'dot(state1, state2) --> state1=state2'
    )
    actions.process(model)
    for a in actions.actions:
        print a
