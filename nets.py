import haiku as hk
import jax.nn as jnn


def initializer(name: str) -> hk.initializers.Initializer:
  return {
      'glorot': hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
      'he': hk.initializers.VarianceScaling(2.0, 'fan_in', 'uniform')
  }[name]


def mlp(
    x,
    activation=jnn.relu,
    initializer=initializer('he'),
    output_sizes=(128, 128, 1),
):
  x = hk.nets.MLP(output_sizes, activation=activation, w_init=initializer)(x)
  return x.squeeze(-1)


def cnn(x, depth, kernels, activation=jnn.relu, initializer=initializer('he')):
  kwargs = {'stride': 2, 'padding': 'VALID', 'w_init': initializer}
  for i, kernel in enumerate(kernels):
    layer_depth = 2**i * depth
    x = activation(hk.Conv2D(layer_depth, kernel, **kwargs)(x))
  return x
