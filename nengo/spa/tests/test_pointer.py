import pytest
import nengo
import numpy as np

from nengo.spa.pointer import SemanticPointer


def test_init():
    a = SemanticPointer([1, 2, 3, 4])
    assert len(a) == 4

    a = SemanticPointer([1, 2, 3, 4, 5])
    assert len(a) == 5

    a = SemanticPointer(range(100))
    assert len(a) == 100

    a = SemanticPointer(27)
    assert len(a) == 27
    assert np.allclose(a.length(), 1)

    with pytest.raises(Exception):
        a = SemanticPointer(np.zeros(2, 2))

    with pytest.raises(Exception):
        a = SemanticPointer(-1)
    with pytest.raises(Exception):
        a = SemanticPointer(0)
    with pytest.raises(Exception):
        a = SemanticPointer(1.7)
    with pytest.raises(Exception):
        a = SemanticPointer(None)
    with pytest.raises(Exception):
        a = SemanticPointer(int)


def test_length():
    a = SemanticPointer([1, 1])
    assert np.allclose(a.length(), np.sqrt(2))


def test_normalize():
    a = SemanticPointer([1, 1])
    a.normalize()
    assert np.allclose(a.length(), 1)


def test_str():
    a = SemanticPointer([1, 1])
    assert str(a) == '[ 1.  1.]'


def test_randomize():
    np.random.seed(0)
    a = SemanticPointer(100)
    std = np.std(a.v)
    assert np.allclose(std, 1.0/np.sqrt(len(a)), rtol=0.01)

    a = SemanticPointer(range(100))
    a.randomize()
    std = np.std(a.v)
    assert np.allclose(std, 1.0/np.sqrt(len(a)), rtol=0.01)


def test_make_unitary():
    np.random.seed(1)
    a = SemanticPointer(100)
    a.make_unitary()
    assert np.allclose(1, a.length(), (a*a).length(), (a*a*a).length())


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
