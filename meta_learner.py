from typing import Tuple

import abc

import jax.numpy as jnp

from haiku import Params


class MetaLearner(abc.ABC):

  # @abc.abstractmethod
  # def __call__(self, x: jnp.ndarray):
  #   """
  #   Predict.
  #   """

  @property
  @abc.abstractmethod
  def prior_params(self) -> Params:
    """
    Return the meta-learner's parameters.
    """

  @prior_params.setter
  @abc.abstractmethod
  def prior_params(self, new_parameters: Params):
    """
    Update the parameters of the meta-learner given new ones.
    """

  @property
  @abc.abstractmethod
  def posterior_params(self) -> Params:
    """
    Return the learner's parameters.
    """

  @posterior_params.setter
  @abc.abstractmethod
  def posterior_params(self, new_parameters: Params):
    """
    Update the parameters given new ones (typically after adaptation).
    """

  @abc.abstractmethod
  def adaptation_step(self, params: Params, x: jnp.ndarray,
                      y: jnp.ndarray) -> Params:
    """
    Adapt to a new task given observations.
    """

  @abc.abstractmethod
  def update_step(self, prior_params: Params, support: Tuple[jnp.ndarray,
                                                       jnp.ndarray],
                  query: Tuple[jnp.ndarray, jnp.ndarray]):
    """
    Compute the grads needed to apply in order to update the model's parameters.
    """