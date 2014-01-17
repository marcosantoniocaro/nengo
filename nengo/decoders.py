"""
This file contains functions concerned with solving for decoders
or other types of weight matrices.

The idea is that as we develop new methods to find decoders either more
quickly or with different constraints (e.g., L1-norm regularization),
the associated functions will be placed here.

Notes
-----

All decoder solvers take following arguments:
  A : array_like (M, N)
    Matrix of the N neurons' activities at the M evaluation points
  Y : array_like (M, D)
    Matrix of the target decoded values for each of the D dimensions,
    at each of the M evaluation points.

All decoder solvers have the following optional keyword parameters:
  rng : numpy.RandomState
    A random number generator to use as required. If none is provided,
    numpy.random will be used.
  E : array_like (D, N2)
    Array of post-population encoders. Providing this tells the solver
    to return an array of connection weights rather than decoders.

All decoder solvers return the following:
  X : np.ndarray (N, D) or (N, N2)
    (N, D) array of decoders if E is none, or (N, N2) array of weights
    if E is not none.

"""

import numpy as np
try:
    import scipy.linalg
    import scipy.optimize
except ImportError:
    scipy = None

try:
    import sklearn.linear_model
except ImportError:
    sklearn = None


def lstsq(A, Y, rng=np.random, E=None, rcond=0.01):
    """Unregularized least-squares"""
    Y = np.dot(Y, E) if E is not None else Y
    X, res, rank, s = np.linalg.lstsq(A, Y, rcond=rcond)
    return X


def lstsq_noise(A, Y, rng=np.random, E=None, noise_amp=0.1):
    sigma = noise_amp * A.max()
    A = A + rng.normal(scale=sigma, size=A.shape)
    Y = np.dot(Y, E) if E is not None else Y
    return _cholesky(A, Y, 0)


def lstsq_multnoise(A, Y, rng=np.random, E=None, noise_amp=0.1):
    A = A + rng.normal(scale=noise_amp, size=A.shape) * A
    Y = np.dot(Y, E) if E is not None else Y
    return _cholesky(A, Y, 0)


def lstsq_L2(A, Y, rng=np.random, E=None, noise_amp=0.1):
    """Least-squares with L2 regularization."""
    Y = np.dot(Y, E) if E is not None else Y
    sigma = noise_amp * A.max()
    return _cholesky(A, Y, sigma)


def lstsq_L2nz(A, Y, rng, E=None, noise_amp=0.1):
    """Least-squares with L2 regularization on non-zero components."""
    Y = np.dot(Y, E) if E is not None else Y

    # Compute the equivalent noise standard deviation. This equals the
    # base amplitude (noise_amp times the overall max activation) times
    # the square-root of the fraction of non-zero components.
    sigma = (noise_amp * A.max()) * np.sqrt((A > 0).mean(0))

    # sigma == 0 means the neuron is never active, so won't be used, but
    # we have to make sigma != 0 for numeric reasons.
    sigma[sigma == 0] = 1

    # Solve the LS problem using the Cholesky decomposition
    return _cholesky(A, Y, sigma)


def lstsq_L1(A, Y, rng, E=None, reg=0.9):
    assert E is not None, "Must provide post encoders"
    assert scipy is not None, "lstsq_L1 requires scipy to be installed"

    ### solve for standard decoders
    x0 = lstsq_L2nz(A, Y, rng)
    w0 = np.dot(x0, E)
    l0 = np.abs(w0).sum()
    xshape = x0.shape

    A = A
    y = Y.flatten()
    x0 = x0.flatten()
    # x0 = np.zeros_like(x0)

    def func(x):
        r = y - np.dot(A, x.reshape(xshape)).flatten()
        return (r**2).sum()

    r0 = func(x0)

    def f_ieqcons(x):
        w = np.dot(x.reshape(xshape), E)
        return reg * l0 - np.abs(w).sum()

    x1, r1, itns, imode, smode = scipy.optimize.fmin_slsqp(
        func, x0, f_ieqcons=f_ieqcons, iter=1000, full_output=True)

    if imode:
        raise Exception(smode)

    x1 = x1.reshape(xshape)
    w1 = np.dot(x1, E)

    np.savez('weights.npz', w0=w0, w1=w1)

    print r0, r1
    return x1


def lstsq_L2_weights(A, Y, rng, E, noise_amp=0.1):
    # TODO: is this a good level of regularization?
    sigma = noise_amp * A.max()

    ### solve least-squares A*X = Y, where Y = Y * encoders
    E = E.T
    A = A
    Y = np.dot(Y, E)

    W = _cholesky(A, Y, sigma)
    return W


def lstsq_L1_weights(A, Y, rng, E, noise_amp=0.1):
    # TODO: is this a good level of regularization?
    # sigma = noise_amp * A.max()

    ### solve least-squares A*X = Y, where Y = Y * encoders
    E = E.T
    A = A
    Y = np.dot(Y, E)

    W0 = _cholesky(A, Y, sigma=1e-3)
    norm0 = np.abs(W0).sum()
    reg = noise_amp * norm0
    # return W

    ### do weight columns one at a time (all at once is too slow)
    ### TODO: does this make a difference in the result?

    Wshape = (A.shape[1], Y.shape[1])
    AA = np.dot(A.T, A)
    # AY = np.dot(A.T, Y)

    W = np.zeros(Wshape)
    for j, y in enumerate(Y.T):
        Ay = np.dot(A.T, y)

        def func(x):
            return ((y - np.dot(A, x))**2).sum()
        def dfunc(x):
            return 2 * (np.dot(AA, x) - Ay)

        def f_ieqcons(x):
            return reg - np.abs(x).sum()
        def df_ieqcons(x):
            return -np.sign(x)

        x0 = np.zeros_like(Ay)
        # x1, r1, itns, imode, smode = scipy.optimize.fmin_slsqp(
        #     func, x0, f_ieqcons=f_ieqcons,
        #     iter=1000, full_output=True, iprint=1)
        x1, r1, itns, imode, smode = scipy.optimize.fmin_slsqp(
            func, x0, f_ieqcons=f_ieqcons,
            fprime=dfunc, fprime_ieqcons=df_ieqcons,
            iter=1000, full_output=True, iprint=1)

        if imode:
            raise Exception(smode)

        W[:, j] = x1

    # W1 = x1.reshape(Wshape)

    # np.savez('weights.npz', w0=w0, w1=w1)

    # print r0, r1
    return W


def lstsq_lasso(A, Y, rng, E, noise_amp=0.01):
    # TODO: is this a good level of regularization?
    # sigma = noise_amp * A.max()

    b = noise_amp * A.max()  # L2 regularization
    a = 0.001 * A.max()      # L1 regularization

    ### solve least-squares A*X = Y, where Y = Y * encoders
    # E = E.T
    A = A
    Y = np.dot(Y, E)

    # lasso = sklearn.linear_model.Lasso(alpha=sigma, fit_intercept=False, max_iter=100000000)
    lasso = sklearn.linear_model.ElasticNet(
        alpha=b, l1_ratio=0.1, fit_intercept=False, max_iter=1000)
    lasso.fit(A, Y)
    W = lasso.coef_.T

    assert W.shape == (A.shape[1], Y.shape[1])
    return W


def _lstsq_drop(A, Y, rng, E, noise_amp, drop):
    """Find coefficients (decoders/weights) with L2 regularization,
    drop those nearest to zero, retrain remaining

    A - A
    Y - Y (not multiplied by encoders)
    """

    # solve for coefficients using standard L2
    X = lstsq_L2nz(A, Y, rng, noise_amp=noise_amp)
    if E is not None:
        X = np.dot(X, E)

    # drop weights close to zero, based on `drop` ratio
    drop = 0.5
    Wabs = np.sort(np.abs(W.flat))
    threshold = Wabs[np.round(drop * Wabs.size)]
    W[np.abs(W) < threshold] = 0

    # retrain nonzero weights
    if E is not None:
        Y = np.dot(Y, E)

    for i in xrange(W.shape[1]):
        nz = W[:,i] != 0
        if nz.sum() > 0:
            W[nz,i] = lstsq_L2nz(A[:,nz], Y[:,i], rng, noise_amp=0.1 * noise_amp)

    return W


def lstsq_drop(A, Y, rng, E, noise_amp=0.1):
    """
    Train with L2 regularization, drop decoders near zero, and retrain
    remaining decoders.
    """

    # solve for weights using standard L2
    decoders = lstsq_L2nz(A, Y, rng, noise_amp=noise_amp)
    W = np.dot(decoders, E.T)

    # drop weights close to zero, based on `drop` ratio
    drop = 0.5
    Wabs = np.sort(np.abs(W.flat))
    threshold = Wabs[np.round(drop * Wabs.size)]
    W[np.abs(W) < threshold] = 0

    # retrain nonzero weights
    A = A
    Y = np.dot(Y, E.T)
    for i in xrange(W.shape[1]):
        nz = W[:,i] != 0
        if nz.sum() > 0:
            W[nz,i] = lstsq_L2nz(A[:,nz], Y[:,i], rng, noise_amp=0.1 * noise_amp)

    return W


def lstsq_drop_weights(A, Y, rng, E, noise_amp=0.1):
    """
    Train with L2 regularization, drop weights near zero, and retrain
    remaining weights.
    """

    # solve for weights using standard L2
    decoders = lstsq_L2nz(A, Y, rng, noise_amp=noise_amp)
    W = np.dot(decoders, E.T)

    # drop weights close to zero, based on `drop` ratio
    drop = 0.5
    Wabs = np.sort(np.abs(W.flat))
    threshold = Wabs[np.round(drop * Wabs.size)]
    W[np.abs(W) < threshold] = 0

    # retrain nonzero weights
    A = A
    Y = np.dot(Y, E.T)
    for i in xrange(W.shape[1]):
        nz = W[:,i] != 0
        if nz.sum() > 0:
            W[nz,i] = lstsq_L2nz(A[:,nz], Y[:,i], rng, noise_amp=0.1 * noise_amp)

    return W


def _cholesky(A, b, sigma, transpose=None):
    """
    Find the least-squares solution of the given linear system(s)
    using the Cholesky decomposition.
    """
    m, n = A.shape
    transpose = m < n if transpose is None else transpose
    if transpose:
        # substitution: x = A'*xbar, G*xbar = b where G = A*A' + lambda*I
        G = np.dot(A, A.T)
    else:
        # multiplication by A': G*x = A'*b where G = A'*A + lambda*I
        G = np.dot(A.T, A)
        b = np.dot(A.T, b)

    reglambda = sigma ** 2 * m  # regularization parameter lambda
    np.fill_diagonal(G, G.diagonal() + reglambda)

    if scipy is not None:
        factor = scipy.linalg.cho_factor(G, overwrite_a=True)
        x = scipy.linalg.cho_solve(factor, b)
    else:
        L = np.linalg.cholesky(G)
        L = np.linalg.inv(L.T)
        x = np.dot(L, np.dot(L.T, b))

    return np.dot(A.T, x) if transpose else x
