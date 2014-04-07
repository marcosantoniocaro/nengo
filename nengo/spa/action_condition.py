
from action_objects import Symbol, Source


class DotProduct(object):
    """The dot product of a Source and a Source or a Source and a Symbol.

    This represents a similarity measure for computing the utility of
    and action.  It also maintains a scaling factor on the result,
    so that the 0.5 in "0.5*DotProduct(Source('vision'), Symbol('DOG'))"
    can be correctly tracked.

    This class is meant to be used with an eval-based parsing system in the
    Condition class, so that the above DotProduct can also be created with
    "0.5*dot(vision,DOG)".
    """
    def __init__(self, item1, item2, scale=1.0):
        if isinstance(item1, (int, float)):
            item1 = Symbol(str(item1))
        if isinstance(item2, (int, float)):
            item2 = Symbol(str(item2))
        if not isinstance(item1, (Source, Symbol)):
            raise TypeError('The first item in the dot product is not a ' +
                            'semantic pointer or a spa.Module output')
        if not isinstance(item2, (Source, Symbol)):
            raise TypeError('The second item in the dot product is not a ' +
                            'semantic pointer or a spa.Module output')
        if not isinstance(item1, Source) and not isinstance(item2, Source):
            raise TypeError('One of the two terms for the dot product ' +
                            'must be a spa.Module output')
        self.item1 = item1
        self.item2 = item2
        self.scale = scale

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return DotProduct(self.item1, self.item2, self.scale * other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if isinstance(other, (int, float)):
            return DotProduct(self.item1, self.item2, self.scale / other)
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, (int, float, DotProduct)):
            return ConditionList([self, other])
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return DotProduct(self.item1, self.item2, -self.scale)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __str__(self):
        if self.scale == 1.0:
            scale_text = ''
        else:
            scale_text = '%g * ' % self.scale
        return '%sdot(%s, %s)' % (scale_text, self.item1, self.item2)



class ConditionList(object):
    """A list of DotProducts and scalars (ints or floats).

    Addition and subtraction extend the list, and multiplication is
    applied to all items in the list.
    """
    def __init__(self, items):
        self.items = items

    def __mul__(self, other):
        return ConditionList([dp*other for dp in self.items])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        return ConditionList([dp/other for dp in self.items])

    def __add__(self, other):
        if isinstance(other, (int, float, DotProduct)):
            return ConditionList(self.items + [other])
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __neg__(self):
        return ConditionList([-dp for dp in self.items])

    def __str__(self):
        return ' + '.join([str(x) for x in self.items])



class Condition(object):
    def __init__(self, sources, condition):
        self.objects = {}
        for name in sources:
            self.objects[name] = Source(name)
        self.objects['dot'] = DotProduct

        condition = condition.replace('\n', ' ')

        self.condition = eval(condition, {}, self)

        if isinstance(self.condition, (int, float, DotProduct)):
            self.condition = ConditionList([self.condition])

    def __getitem__(self, key):
        item = self.objects.get(key, None)
        if item is None:
            if not key[0].isupper():
                raise KeyError('Semantic pointers must begin with a capital')
            item = Symbol(key)
            self.objects[key] = item
        return item

    def __str__(self):
        return str(self.condition)
