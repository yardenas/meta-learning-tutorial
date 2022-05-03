import pytest

from omniglot_dataset import Omniglot


@pytest.fixture
def omniglot_dataset():
  # 128 Tasks, three-way, one-shot.
  return Omniglot(128, 3, 1)


def test_make_dataset(omniglot_dataset):
  support, query = next(iter(omniglot_dataset.train_set))
  assert support[0].shape == (128, 3, 28, 28, 1)
  assert support[1].shape == (128, 3, 3)
