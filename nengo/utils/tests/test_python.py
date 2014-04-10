import logging
import pytest

from nengo.utils.python import partial

logger = logging.getLogger(__name__)


def test_partial():
    def f0(a, b, c=3, d=4):
        """my docstring"""
        return a, b, c, d

    f1 = partial(f0, d=5)
    f2 = partial(f0, 1, 2)

    for attr in ['__name__', '__doc__']:
        assert getattr(f0, attr) == getattr(f1, attr)
        assert getattr(f0, attr) == getattr(f2, attr)

    assert f0(1, 2) == (1, 2, 3, 4)
    assert f1(1, 2) == (1, 2, 3, 5)
    assert f0(1, 2, c=2) == (1, 2, 2, 4)
    assert f1(1, 2, c=2) == (1, 2, 2, 5)
    assert f0(1, 2, c=2, d=1) == (1, 2, 2, 1)
    with pytest.raises(TypeError):
        f1(1, 2, c=2, d=4)

    assert f0(1, 2) == f2()
    assert f0(1, 2, c=2) == f2(c=2)
    assert f0(1, 2, c=2, d=1) == f2(c=2, d=1)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
