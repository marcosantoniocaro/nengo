import copy

import numpy as np

import nengo
from nengo.utils.network import with_self


class EnsembleArray(nengo.Network):

    def __init__(self, neurons, n_ensembles, ens_dimensions=1, label=None,
                 **ens_kwargs):
        if "dimensions" in ens_kwargs:
            raise TypeError(
                "'dimensions' is not a valid argument to EnsembleArray. "
                "To set the number of ensembles, use 'n_ensembles'. To set "
                "the number of dimensions per ensemble, use 'ens_dimensions'.")

        label_prefix = "" if label is None else label + "_"

        self.n_ensembles = n_ensembles
        self.dimensions_per_ensemble = ens_dimensions
        transform = np.eye(self.dimensions)

        self.input = nengo.Node(size_in=self.dimensions, label="input")

        for i in range(n_ensembles):
            e = nengo.Ensemble(
                copy.deepcopy(neurons), self.dimensions_per_ensemble,
                label=label_prefix + str(i), **ens_kwargs)
            trans = transform[i * self.dimensions_per_ensemble:
                              (i + 1) * self.dimensions_per_ensemble, :]
            nengo.Connection(self.input, e, transform=trans, synapse=None)

        self.add_output('output', function=None)

    @with_self
    def add_output(self, name, function, synapse=None, **conn_kwargs):
        if function is None:
            function_d = self.dimensions_per_ensemble
        else:
            func_output = function(np.zeros(self.dimensions_per_ensemble))
            function_d = np.asarray(func_output).size

        dim = self.n_ensembles * function_d
        output = nengo.Node(size_in=dim, label=name)
        setattr(self, name, output)

        for i, e in enumerate(self.ensembles):
            nengo.Connection(
                e, output[i*function_d:(i+1)*function_d], function=function,
                synapse=synapse, **conn_kwargs)
        return output

    @property
    def dimensions(self):
        return self.n_ensembles * self.dimensions_per_ensemble
