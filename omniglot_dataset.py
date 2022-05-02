from typing import Iterator, List, Tuple

import os
import random

import numpy as np

from tensorflow import data as tfd
from tensorflow import image as tfi
from tensorflow import io as tfio
from tensorflow import dtypes
import tensorflow as tf

from google_drive_downloader import GoogleDriveDownloader


class Omniglot:

  def __init__(self,
               meta_batch_size: int,
               num_classes: int,
               num_samples_per_class: int,
               seed: int = 666):
    self.meta_batch_size = meta_batch_size
    self.num_samples_per_class = num_samples_per_class
    self.num_classes = num_classes
    self.seed = seed
    if not os.path.isdir('./omniglot_resized'):
      GoogleDriveDownloader.download_file_from_google_drive(
          file_id='1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI',
          dest_path='./omniglot_resized.zip',
          unzip=True)

    data_folder = './omniglot_resized'
    self.img_size = 28, 28

    character_folders = [
        os.path.join(data_folder, family, character)
        for family in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, family))
        for character in os.listdir(os.path.join(data_folder, family))
        if os.path.isdir(os.path.join(data_folder, family, character))
    ]

    random.seed(1)
    random.shuffle(character_folders)
    num_val = 100
    num_train = 1100
    self.metatrain = self._make_dataset(character_folders[:num_train])
    self.metaval = self._make_dataset(character_folders[num_train:num_train +
                                                        num_val])
    self.metatest = self._make_dataset(character_folders[num_train + num_val:])

  @property
  def train_set(
      self
  ) -> Iterator[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray,
                                                           np.ndarray]]]:
    yield from self.metatrain.as_numpy_iterator()

  @property
  def eval_set(
      self
  ) -> Iterator[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray,
                                                           np.ndarray]]]:
    yield from self.metaval.as_numpy_iterator()

  @property
  def test_set(
      self
  ) -> Iterator[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray,
                                                           np.ndarray]]]:
    yield from self.metatest.as_numpy_iterator()

  def _make_dataset(self, folders: List[str]) -> tfd.Dataset:
    characters = tfd.Dataset.from_tensor_slices(folders).shuffle(
        1100, seed=self.seed, reshuffle_each_iteration=True)

    def get_images_filenames(char):
      all_images = tfio.matching_files(char + '/*.png')
      return tfd.Dataset.from_tensor_slices(
          tf.random.shuffle(all_images,
                            seed=self.seed)[:self.num_samples_per_class + 1])

    # Use interleave to read the relevant .png files as we iterate through the
    # 1100 different chars. Set block_length to num_samples_per_class so that
    # we can next batch images from same char together.
    image_filenames = characters.interleave(
        get_images_filenames,
        num_parallel_calls=tfd.AUTOTUNE,
        block_length=self.num_samples_per_class + 1)

    def load_image(image_filename):
      img = tfio.read_file(image_filename)
      img = tfio.decode_png(img, channels=1)
      img = tfi.resize(img, self.img_size)
      img = tf.cast(img, dtypes.float32) / 255.0
      img = 1.0 - img
      return img

    # Unbatch map and batch to allow tf to read images concurrently. Class
    # grouping is maintained.
    shots = image_filenames.map(
        load_image,
        num_parallel_calls=tfd.AUTOTUNE).batch(self.num_samples_per_class + 1)
    ways = shots.batch(self.num_classes)
    tasks = ways.batch(self.meta_batch_size)

    def to_support_and_query_sets(batch):
      support_x, query_x = tf.split(
          tf.transpose(batch, (0, 2, 1, 3, 4, 5)),
          (self.num_samples_per_class, 1),
          axis=1)
      support_y, query_y = tf.split(
          tf.eye(
              self.num_classes,
              batch_shape=(self.meta_batch_size,
                           self.num_samples_per_class + 1)),
          (self.num_samples_per_class, 1),
          axis=1)
      ids = tf.range(0, self.num_classes, dtype=dtypes.int32)
      ids = tf.random.shuffle(ids, seed=self.seed)
      query_x = tf.gather(query_x, ids, axis=2)
      query_y = tf.gather(query_y, ids, axis=2)
      return (support_x, support_y), (query_x, query_y)

    return tasks.map(
        to_support_and_query_sets,
        num_parallel_calls=tfd.AUTOTUNE).prefetch(tfd.AUTOTUNE)
