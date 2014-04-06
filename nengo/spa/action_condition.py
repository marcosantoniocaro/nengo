
from action_objects import Symbol, Source


class Scalar(object):
    def __init__(self, scalar):
        self.scalar = scalar

    def __str__(self):
        return '%g' % self.scalar


class DotProduct(Scalar):
    def __init__(self, item1, item2, scale):
        self.item1 = item1
        self.item2 = item2
        self.scale = 1.0

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return DotProduct(self.item1, self.item2, self.scale * other)
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)
        if isinstance(other, Scalar):
            return ScalarList([self, other])
        else:
            return NotImplemented

    def __neg__(self):
        return DotProduct(self.item1, self.item2, -self.scale)

    def __sub__(self, other):
        return self + (-other)

    def __div__(self, other):
        if isinstance(other, (int, float)):
            return DotProduct(self.item1, self.item2, self.scale / other)
        else:
            return NotImplemented

    def __str__(self):
        if self.scale == 1.0:
            scale_text = ''
        else:
            scale_text = '%g * ' % self.scale
        return '%sdot(%s, %s)' % (scale_text, self.item1, self.item2)


class ScalarList(object):
    def __init__(self, items):
        self.items = items

    def __mul__(self, other):
        return ScalarList([dp*other for dp in self.items])
    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        return ScalarList([dp/other for dp in self.items])

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)
        if isinstance(other, Scalar):
            return ScalarList(self.items + [other])
        else:
            return NotImplemented

    def __str__(self):
        return ' + '.join([str(x) for x in self.items])


def dot(a, b):
    if not isinstance(a, (Source, Symbol)):
        raise TypeError('Cannot combine non-vector items')
    if not isinstance(b, (Source, Symbol)):
        raise TypeError('Cannot combine non-vector items')
    if not isinstance(a, Source) and not isinstance(b, Source):
        raise TypeError('Must do DotProduct on at least one source')
    return DotProduct(a, b, 1.0)


class Condition(object):
    def __init__(self, sources, condition):
        self.objects = {}
        for name in sources:
            self.objects[name] = Source(name)
        self.objects['dot'] = dot


        condition = condition.replace('\n', ' ')
        print condition

        result = eval(condition, {}, self)

        if isinstance(result, (int, float)):
            result = ScalarList([Scalar(result)])
        elif isinstance(result, Scalar):
            result = ScalarList([result])
        self.condition = result

    def __getitem__(self, key):
        item = self.objects.get(key, None)
        if item is None:
            item = Symbol(key)
            self.objects[key] = item
        return item

    def __str__(self):
        return str(self.condition)


if __name__ == '__main__':
    c = Condition(['state1', 'state2'],
                  'dot(A*(B+C+2*D)*state1,Q)+0.5+dot(state2, state1)')
    print c.condition
