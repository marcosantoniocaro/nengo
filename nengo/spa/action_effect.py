
from action_objects import Symbol, Source


class SourceWithAddition(Source):
    def __add__(self, other):
        if isinstance(other, (Symbol, Source)):
            return VectorList([self, other])
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return -self.__add__(other)


class VectorList(object):
    def __init__(self, items):
        self.items = items

    def __mul__(self, other):
        return VectorList([x*other for x in self.items])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Symbol('%g' % other)
        if isinstance(other, (Symbol, Source)):
            return VectorList(self.items + [other])
        elif isinstance(other, VectorList):
            return VectorList(self.items + other.items)
        else:
            return NotImplemented

    def __sub__(self, other):
        return self.__add__(-other)

    def __neg__(self):
        return VectorList([-x for x in self.items])

    def __str__(self):
        return ' + '.join([str(v) for v in self.items])


class Effect(object):
    def __init__(self, sources, effect):
        self.objects = {}
        for name in sources:
            self.objects[name] = SourceWithAddition(name)
        self.objects['__effect_dictionary'] = dict

        self.effect = eval('__effect_dictionary(%s)' % effect, {}, self)
        for k, v in self.effect.items():
            if isinstance(v, (Symbol, Source)):
                self.effect[k] = VectorList([v])

    def __getitem__(self, key):
        item = self.objects.get(key, None)
        if item is None:
            if not key[0].isupper():
                raise KeyError('Semantic pointers must begin with a capital')
            item = Symbol(key)
            self.objects[key] = item
        return item

    def __str__(self):
        return ', '.join(['%s=%s' % x for x in self.effect.items()])


if __name__ == '__main__':
    e = Effect(['state1', 'state2'], 'motor=A')
    print e
    e = Effect(['state1', 'state2'], 'motor=A*B+C')
    print e
    e = Effect(['state1', 'state2'], 'motor=state1')
    print e
    e = Effect(['state1', 'state2'], 'motor=state1*A')
    print e
    e = Effect(['state1', 'state2'], 'motor=state1*A+B')
    print e
    e = Effect(['state1', 'state2'], 'motor=C*(state1*(A-D)+B)-.5*Q')
    print e
    e = Effect(['state1', 'state2'], 'motor=-state2-(-state1)')
    print e
    e = Effect(['state1', 'state2'], 'motor=-A*(state2-A)')
    print e
