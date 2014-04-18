import nengo.config
import nengo.objects
from nengo.config import Default


@nengo.config.configures(nengo.objects.Ensemble)
class EmptyEnsemble(nengo.config.ConfigItem):
    radius = Default
    encoders = Default
    intercepts = Default
    max_rates = Default
    eval_points = Default
    seed = Default
    label = Default


@nengo.config.configures(nengo.objects.Node)
class EmptyNode(nengo.config.ConfigItem):
    output = Default
    size_in = Default
    size_out = Default
    label = Default


@nengo.config.configures(nengo.objects.Connection)
class EmptyConnection(nengo.config.ConfigItem):
    synapse = Default
    transform = Default
    modulatory = Default
    weight_solver = Default
    decoder_solver = Default
    function = Default
    eval_points = Default


@nengo.config.configures(nengo.objects.Neurons)
class EmptyNeurons(nengo.config.ConfigItem):
    bias = Default
    gain = Default
    label = Default
