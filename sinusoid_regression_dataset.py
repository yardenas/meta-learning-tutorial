from typing import Iterator, Tuple

import numpy as np


class SinusoidRegression:

  def __init__(self, meta_batch_size: int, num_shots: int, seed: int = 666):
    self.meta_batch_size = meta_batch_size
    self.num_shots = num_shots
    self.rs = np.random.RandomState(seed)

  @property
  def train_set(
      self
  ) -> Iterator[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray,
                                                           np.ndarray]]]:
    while True:
      yield self._make_batch()

  @property
  def eval_set(
      self
  ) -> Iterator[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray,
                                                           np.ndarray]]]:
    while True:
      yield self._make_batch()

  @property
  def test_set(
      self
  ) -> Iterator[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray,
                                                           np.ndarray]]]:
    while True:
      yield self._make_batch()

  def _make_batch(
      self
  ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    # Select amplitude and phase for the task
    amplitudes = []
    phases = []
    for _ in range(self.meta_batch_size):
      amplitudes.append(self.rs.uniform(low=0.1, high=.5))
      phases.append(self.rs.uniform(low=0., high=np.pi))

    def get_batch():
      xs, ys = [], []
      for amplitude, phase in zip(amplitudes, phases):
        x = self.rs.uniform(low=-5., high=5., size=(self.num_shots, 1))
        y = amplitude * np.sin(x + phase)
        xs.append(x)
        ys.append(y)
      return np.stack(xs), np.stack(ys)

    x1, y1 = get_batch()
    x2, y2 = get_batch()
    return (x1, y1), (x2, y2)
