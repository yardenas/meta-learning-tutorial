import pytest

from omniglot_dataset import Omniglot


@pytest.fixture
def omniglot_dataset():
  return Omniglot(128, 3, 1)


def test_make_dataset(omniglot_dataset):
  images, labels = next(
      iter(
          omniglot_dataset._make_dataset(
              omniglot_dataset.metatrain_character_folders)))
  assert tuple(images.numpy().shape) == (128, 2, 3, 28, 28, 1)
  assert tuple(labels.numpy().shape) == (128, 2, 3, 3)
