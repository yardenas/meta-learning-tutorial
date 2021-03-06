import functools
from typing import Callable, Tuple

from functools import partial

import jax
import jax.numpy as jnp
from haiku import Params
import haiku as hk

from meta_learner import MetaLearner


class Maml(MetaLearner):

  def __init__(self,
               model: Callable,
               example: jnp.ndarray,
               inner_lr: float,
               adaptation_steps: int = 1):
    net = hk.without_apply_rng(hk.transform(lambda x: model(x)))
    self.net = net.apply
    self._prior_params = net.init(jax.random.PRNGKey(666), example)
    self._inner_lr = inner_lr
    self._adaptation_steps = adaptation_steps

  @functools.partial(jax.jit, static_argnums=0)
  def __call__(self, params: Params, x: jnp.ndarray):
    return self.net(params, x)

  @property
  def prior_params(self) -> Params:
    return self._prior_params

  @prior_params.setter
  def prior_params(self, new_parameters: Params):
    self._prior_params = new_parameters

  def adaptation_step(self, params: Params, x: jnp.ndarray, y: jnp.ndarray):
    return self._adapt(params, x, y)

  @partial(jax.jit, static_argnums=0)
  @partial(jax.vmap, in_axes=[None, None, 0, 0])
  def _adapt(self, params: Params, x: jnp.ndarray, y: jnp.ndarray):
    new_params = params

    def loss(params: Params):
      log_likelihood = self.net(params, x).log_prob(y).mean()
      return -log_likelihood

    for _ in range(self._adaptation_steps):
      grads = jax.grad(loss)(new_params)
      new_params = jax.tree_map(lambda p, g: p - self._inner_lr * g, new_params,
                                grads)
    return new_params

  def update_step(self, prior_params: Params, support: Tuple[jnp.ndarray,
                                                             jnp.ndarray],
                  query: Tuple[jnp.ndarray, jnp.ndarray]):
    return self._maml_step(prior_params, support, query)

  @partial(jax.jit, static_argnums=0)
  def _maml_step(self, params: Params, support: Tuple[jnp.ndarray, jnp.ndarray],
                 query: Tuple[jnp.ndarray, jnp.ndarray]):

    def inner_step(params):
      x_support, y_support = support
      posterior_params = self._adapt(params, x_support, y_support)
      x_query, y_query = query
      log_likelihood_fn = jax.vmap(lambda p, x, y: self.net(p, x).log_prob(y))
      log_likelihood = log_likelihood_fn(posterior_params, x_query,
                                         y_query).mean()
      return -log_likelihood

    return jax.grad(inner_step)(params)
