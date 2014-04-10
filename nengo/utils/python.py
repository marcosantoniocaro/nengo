
def partial(func, *pargs, **pkwargs):
    """Implementation of `functools.partial` that keeps __name__ and __doc__.

    Based off the equivalent example code for `functools.partial` at
        https://docs.python.org/3.3/library/functools.html
    """
    name = str(func.__name__)

    def newfunc(*fargs, **fkwargs):
        args = pargs + fargs
        kwargs = pkwargs.copy()
        for k, v in fkwargs.items():
            if k in kwargs:
                raise TypeError(
                    "%s got multiple values for the keyword argument '%s'" %
                    (name, k))
            else:
                kwargs[k] = v
        return func(*args, **kwargs)

    newfunc.func = func
    newfunc.args = pargs
    newfunc.keywords = pkwargs
    newfunc.__name__ = func.__name__
    newfunc.__doc__ = func.__doc__
    return newfunc
