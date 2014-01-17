import logging

import numpy as np
import pytest

import nengo
from nengo.objects import Uniform
from nengo.helpers import piecewise, whitenoise
# from nengo.helpers import piecewise
from nengo.tests.helpers import Plotter, rms
# from nengo.decoders import _cholesky, lstsq_old, lstsq_noise, lstsq_L2, lstsq_L2nz, lstsq_L1
# from nengo.decoders import lstsq_L2_weights, lstsq_L1_weights
from nengo.decoders import *
import time

logger = logging.getLogger(__name__)


def test_cholesky(Simulator):
    rng = np.random.RandomState(4829)

    m, n = 100, 100
    A = rng.normal(size=(m, n))
    b = rng.normal(size=(m, ))

    x0, _, _, _ = np.linalg.lstsq(A, b)
    x1 = _cholesky(A, b, 0, transpose=False)
    x2 = _cholesky(A, b, 0, transpose=True)

    assert np.allclose(x0, x1)
    assert np.allclose(x0, x2)


def test_weights(Simulator):

    # rng = np.random.RandomState(39408)
    rng = np.random.RandomState()

    d = 2
    m, n = 100, 101
    n_samples = 1000

    a = nengo.LIF(m)
    a.set_gain_bias(rng.uniform(50, 100, m), rng.uniform(-1, 1, m))

    b = nengo.LIF(n)
    b.set_gain_bias(rng.uniform(50, 100, n), rng.uniform(-1, 1, n))

    e1 = np.random.randn(d, m)
    e1 /= np.sqrt((e1**2).sum(axis=0, keepdims=1))
    e2 = np.random.randn(d, n)
    e2 /= np.sqrt((e2**2).sum(axis=0, keepdims=1))

    p1 = nengo.decoders.sample_hypersphere(d, n_samples, rng)

    A = a.rates(np.dot(p1, e1))
    X = p1

    D = lstsq_L2(A, X, rng, noise_amp=0.1)
    W1 = np.dot(D, e2)

    W2 = lstsq_L2_weights(A, X, rng, e2.T, noise_amp=0.1)

    p2 = nengo.decoders.sample_hypersphere(d, n_samples, rng)
    A2 = a.rates(np.dot(p2, e1))

    Y1 = np.dot(A2, W1)
    Y2 = np.dot(A2, W2)
    assert np.allclose(Y1, Y2)
    assert np.allclose(W1, W2)

    # print W1.min(), W1.max()
    # print W2.min(), W2.max()


@pytest.mark.benchmark
def test_solvers(Simulator, nl_nodirect):

    N = 10
    # solvers = [lstsq_noise, lstsq_L2, lstsq_L2nz]
    solvers = [lstsq_old, lstsq_noise, lstsq_L2, lstsq_L2nz]

    dt = 1e-3
    tfinal = 4

    # input_function = whitenoise(step=0.1, high=1)
    # input_function = piecewise({0: -1, 1: -1, 3: 1})

    def input_function(t):
        return np.interp(t, [1, 3], [-1, 1], left=-1, right=1)

    # model = nengo.Model('test_solvers', seed=290)
    model = nengo.Model('test_solvers')
    source = nengo.Node(output=input_function)
    a = nengo.Ensemble(nl_nodirect(N), dimensions=1)
    nengo.Connection(source, a)

    source_p = nengo.Probe(source)
    # a_p = nengo.Probe(a)
    probes = [nengo.Probe(a, decoder_solver=s) for s in solvers]
    probes_f = [nengo.Probe(a, decoder_solver=s, filter=0.03) for s in solvers]
    # probes = [nengo.Probe(a) for s in solvers]
    # probes = [nengo.Probe(a), nengo.Probe(a)]

    sim = nengo.Simulator(model, dt=dt)
    sim.run(tfinal)
    t = sim.trange()

    ref = sim.data(source_p)
    names = [solver.__name__ for solver in solvers]
    outputs = [sim.data(probe) for probe in probes]
    outputs_f = [sim.data(probe) for probe in probes_f]

    max_delay = int(0.02 / dt)
    t_delay = dt * np.arange(max_delay)
    def rmsgram(a, b):
        return np.asarray([rms(a[:len(a)-d] - b[d:]) for d in xrange(max_delay)])

    rmsgrams = [rmsgram(ref, output) for output in outputs_f]
    # rmsf_values = [rms(output - ref) for output in outputs_f]

    buf = 0.1
    rmse_buf = lambda a, b: rms(a[t > buf] - b[t > buf])
    rms_values = [rmse_buf(output, ref) for output in outputs]
    rmsf_values = [rmse_buf(output, ref) for output in outputs_f]

    print
    print nl_nodirect.__name__
    print rms_values
    print rmsf_values


    with Plotter(Simulator, nl_nodirect) as plt:

        subplot_index = [0]
        def subplot():
            subplot_index[0] += 1
            plt.subplot(2, 1, subplot_index[0])

        subplot()
        plt.plot(t, ref, 'k--')
        for name, output in zip(names, outputs_f):
            plt.plot(t, output, label=name)

        subplot()
        for name, output in zip(names, outputs_f):
            plt.plot(t[t > buf], (output - ref)[t > buf], label=name)

        # for name, rmsgram in zip(names, rmsgrams):
        #     plt.plot(t_delay, rmsgram, label=name)

        plt.savefig('test_decoders.test_solvers.pdf')
        plt.close()


@pytest.mark.benchmark
def test_regularization(Simulator, nl_nodirect):

    ### TODO: multiple trials per parameter set, with different seeds

    solvers = [lstsq_L2, lstsq_L2nz]
    neurons = np.array([10, 20, 50, 100])
    regs = np.linspace(0.01, 0.3, 16)
    filters = np.linspace(0, 0.03, 11)

    buf = 0.2  # buffer for initial transients
    dt = 1e-3
    tfinal = 3 + buf

    def input_function(t):
        return np.interp(t, [1, 3], [-1, 1], left=-1, right=1)

    # model = nengo.Model('test_solvers', seed=290)
    model = nengo.Model('test_regularization')
    source = nengo.Node(output=input_function)
    source_p = nengo.Probe(source)

    probes = np.zeros(
        (len(solvers), len(neurons), len(regs), len(filters)),
         dtype='object')

    for j, n_neurons in enumerate(neurons):
        a = nengo.Ensemble(nl_nodirect(n_neurons), dimensions=1)
        nengo.Connection(source, a)

        for i, solver in enumerate(solvers):
            for k, reg in enumerate(regs):
                reg_solver = lambda a, t, r, reg=reg: solver(a, t, r, noise_amp=reg)
                for l, filter in enumerate(filters):
                    probes[i, j, k, l] = nengo.Probe(
                        a, decoder_solver=reg_solver, filter=filter)

    sim = nengo.Simulator(model, dt=dt)
    sim.run(tfinal)
    t = sim.trange()

    ref = sim.data(source_p)
    rmse_buf = lambda a, b: rms(a[t > buf] - b[t > buf])
    rmses = np.zeros(probes.shape)
    for i, probe in enumerate(probes.flat):
        rmses.flat[i] = rmse_buf(sim.data(probe), ref)
    rmses = rmses - rmses[:, :, [0], :]

    with Plotter(Simulator, nl_nodirect) as plt:
        plt.figure(figsize=(8, 12))
        X, Y = np.meshgrid(filters, regs)

        for i, solver in enumerate(solvers):
            for j, n_neurons in enumerate(neurons):
                plt.subplot(len(neurons), len(solvers), len(solvers)*j + i + 1)
                Z = rmses[i, j, :, :]
                plt.contourf(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 21))
                plt.xlabel('filter')
                plt.ylabel('reg')
                plt.title("%s (N=%d)" % (solver.__name__, n_neurons))

        plt.tight_layout()
        plt.savefig('test_decoders.test_regularization.pdf')
        plt.close()


def test_L1(Simulator, nl_nodirect):

    N = 100
    D = 2

    buf = 0.2  # buffer for initial transients
    dt = 1e-3
    tfinal = 3 + buf

    def input_function(t):
        return np.interp(t, [1, 3], [-1, 1], left=-1, right=1) * np.ones(D)

    post_encoders = np.random.randn(D, N)
    post_encoders /= np.sqrt((post_encoders**2).sum(axis=0, keepdims=1))

    model = nengo.Model('test_L1')
    source = nengo.Node(output=input_function)
    a = nengo.Ensemble(nl_nodirect(N), dimensions=D)
    b = nengo.Ensemble(nl_nodirect(N), dimensions=D, encoders=post_encoders.T)
    nengo.Connection(source, a)
    solver = lambda a, t, r, e=post_encoders: lstsq_L1(a, t, r, post_encoders=e)
    c = nengo.Connection(a, b, decoder_solver=solver)

    source_p = nengo.Probe(source)
    a_p = nengo.Probe(a, filter=0.03)
    b_p = nengo.Probe(b, filter=0.03)

    sim = nengo.Simulator(model, dt=dt)
    sim.run(tfinal)
    t = sim.trange()

    ref = sim.data(a_p)
    out = sim.data(b_p)

    data = np.load('weights.npz')
    w0, w1 = data['w0'], data['w1']
    print w0

    with Plotter(Simulator, nl_nodirect) as plt:
        plt.subplot(2, 1, 1)
        plt.plot(t, ref, 'k')
        plt.plot(t, out)

        plt.subplot(2, 2, 3)
        plt.hist(w0, bins=20)

        plt.subplot(2, 2, 4)
        plt.hist(w1, bins=20)

        plt.savefig('test_decoders.test_L1.pdf')
        plt.close()


def test_L1_weights(Simulator, nl_nodirect):

    N = 100
    D = 1

    buf = 0.2  # buffer for initial transients
    dt = 1e-3
    tfinal = 3 + buf

    def input_function(t):
        return np.interp(t, [1, 3], [-1, 1], left=-1, right=1) * np.ones(D)

    seeda = 439
    seedb = seeda + 1

    model = nengo.Model('test_L1')
    source = nengo.Node(output=input_function)
    bargs = dict(seed=seedb,
                 intercepts=Uniform(-0.9, 0.9), max_rates=Uniform(50, 100))
    a1 = nengo.Ensemble(nl_nodirect(N), dimensions=D, seed=seeda)
    b1 = nengo.Ensemble(nl_nodirect(N + 1), dimensions=D, **bargs)
    a2 = nengo.Ensemble(nl_nodirect(N), dimensions=D, seed=seeda)
    b2 = nengo.Ensemble(nl_nodirect(N + 1), dimensions=D, **bargs)

    nengo.Connection(source, a1)
    nengo.Connection(source, a2)

    solver1 = lambda a, t, r: lstsq_L2(a, t, r, noise_amp=0.1)
    c1 = nengo.Connection(a1, b1, decoder_solver=solver1)

    # solver2 = lambda a, t, r, e: lstsq_L2_weights(a, t, r, e, noise_amp=0.1)
    # solver2 = lambda a, t, r, e: lstsq_L1_weights(a, t, r, e, noise_amp=0.1)
    # solver2 = lambda a, t, r, e: lstsq_lasso(a, t, r, e, noise_amp=0.01)
    solver2 = lambda a, t, r, e: lstsq_L2_drop_weights(a, t, r, e, noise_amp=0.1)
    c2 = nengo.Connection(a2, b2, weight_solver=solver2)

    source_p = nengo.Probe(source)
    a1_p = nengo.Probe(a1, filter=0.03)
    b1_p = nengo.Probe(b1, filter=0.03)
    a2_p = nengo.Probe(a2, filter=0.03)
    b2_p = nengo.Probe(b2, filter=0.03)

    sim = nengo.Simulator(model, dt=dt)
    sim.run(tfinal)
    t = sim.trange()

    ref = sim.data(a1_p)
    out1 = sim.data(b1_p)
    out2 = sim.data(b2_p)

    D1 = sim.model.memo[id(c1)]._decoders
    W1 = np.dot(D1, sim.model.memo[id(b1)]._scaled_encoders.T)
    W2 = sim.model.memo[id(c2)]._decoders

    std = np.array([W1, W2]).std()
    bins = np.linspace(-3*std, 3*std, 21)

    nonzero = lambda W: (np.abs(W) > 1e-10).sum() / float(W.size)
    print "W1 nonzero:", nonzero(W1)
    print "W2 nonzero:", nonzero(W2)

    with Plotter(Simulator, nl_nodirect) as plt:
        plt.subplot(2, 1, 1)
        plt.plot(t, ref, 'k')
        plt.plot(t, out1)
        plt.plot(t, out2)

        plt.subplot(2, 2, 3)
        plt.hist(W1.flatten(), bins=bins)

        plt.subplot(2, 2, 4)
        plt.hist(W2.flatten(), bins=bins)

        plt.savefig('test_decoders.test_L2_weights.pdf')
        plt.close()



# def test_L1_weights(Simulator, nl_nodirect):

#     N = 100
#     D = 1

#     buf = 0.2  # buffer for initial transients
#     dt = 1e-3
#     tfinal = 3 + buf

#     def input_function(t):
#         return np.interp(t, [1, 3], [-1, 1], left=-1, right=1)

#     post_encoders = np.random.randn(D, N)
#     post_encoders /= np.sqrt((post_encoders**2).sum(axis=0, keepdims=1))

#     model = nengo.Model('test_L1')
#     source = nengo.Node(output=input_function)
#     a = nengo.Ensemble(nl_nodirect(N), dimensions=D)
#     b = nengo.Ensemble(nl_nodirect(N), dimensions=D, encoders=post_encoders.T)
#     nengo.Connection(source, a)
#     solver = lambda a, t, r, e=post_encoders: lstsq_L1(a, t, r, post_encoders=e)
#     c = nengo.Connection(a, b, decoder_solver=solver)

#     source_p = nengo.Probe(source)
#     a_p = nengo.Probe(a, filter=0.03)
#     b_p = nengo.Probe(b, filter=0.03)

#     sim = nengo.Simulator(model, dt=dt)
#     sim.run(tfinal)
#     t = sim.trange()

#     # import pdb; pdb.set_trace()

#     ref = sim.data(a_p)
#     out = sim.data(b_p)

#     data = np.load('weights.npz')
#     w0, w1 = data['w0'], data['w1']
#     print w0

#     with Plotter(Simulator, nl_nodirect) as plt:
#         plt.subplot(2, 1, 1)
#         plt.plot(t, ref)
#         plt.plot(t, out)

#         plt.subplot(2, 2, 3)
#         plt.hist(w0, bins=20)

#         plt.subplot(2, 2, 4)
#         plt.hist(w1, bins=20)

#         plt.savefig('test_decoders.test_L1.pdf')
#         plt.close()



if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
