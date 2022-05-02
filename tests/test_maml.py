import functools

import pytest
import jax
import haiku as hk
import optax
from tensorflow_probability.substrates import jax as tfp

from omniglot_dataset import Omniglot
import maml
import nets

tfd = tfp.distributions

NUM_CLASSSES = 5
NUM_SHOTS = 1


@pytest.fixture
def dataset():
  return Omniglot(32, NUM_CLASSSES, NUM_SHOTS)


@pytest.fixture
def maml_model(dataset):

  def net(x):

    def cnn(x):
      x = nets.cnn(x, depth=32, kernels=(4, 4, 4))
      x = hk.Flatten()(x)
      logits = hk.Linear(NUM_CLASSSES)(x)
      return logits

    cnn = hk.BatchApply(cnn, 2)
    logits = cnn(x)
    return tfd.OneHotCategorical(logits)

  support, _ = next(dataset.train_set)
  model = maml.Maml(hk.BatchApply(net, 2), support[0], 0.4)
  return model


def test_update(maml_model, dataset):
  opt = optax.flatten(optax.adam(1e-3))
  opt_state = opt.init(maml_model.prior_params)
  maml_model.prior_params, opt_state = update(maml_model, opt,
                                              maml_model.prior_params,
                                              opt_state,
                                              *next(dataset.train_set))


@functools.partial(jax.jit, static_argnums=[0, 1])
def update(model, optimizer, prior_params, opt_state, support, query):
  grads = model.update_step(prior_params, support, query)
  updates, new_opt_state = optimizer.update(grads, opt_state)
  new_params = optax.apply_updates(prior_params, updates)
  return new_params, new_opt_state
