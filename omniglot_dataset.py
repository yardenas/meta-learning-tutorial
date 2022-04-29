from typing import Iterator, List

import numpy as np
import os
import random

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

    self.dim_input = np.prod(self.img_size)
    self.dim_output = self.num_classes

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
    self.metatrain_character_folders = character_folders[:num_train]
    self.metaval_character_folders = character_folders[num_train:num_train +
                                                       num_val]
    self.metatest_character_folders = character_folders[num_train + num_val:]

  def train_sample(self) -> Iterator[np.ndarray]:
    pass

  def eval_sample(self) -> Iterator[np.ndarray]:
    pass

  def test_sample(self) -> Iterator[np.ndarray]:
    pass

  def _make_dataset(self, folders: List[str]) -> tfd.Dataset:
    characters = tfd.Dataset.from_tensor_slices(folders).shuffle(1100)

    def get_images_filenames(char):
      all_images = tfio.matching_files(char + '/*.png')
      return tf.random.shuffle(all_images)[:self.num_samples_per_class + 1]

    image_filenames = characters.map(get_images_filenames)

    def load_image(image_filename):
      img = tfio.read_file(image_filename)
      img = tfio.decode_png(img, channels=1)
      img = tfi.resize(img, self.img_size)
      img = tf.cast(img, dtypes.float32) / 255.0
      img = 1.0 - img
      return img

    # Unbatch map and batch to allow tf to read images concurrently. Class
    # grouping is maintained.
    shots = image_filenames.unbatch().map(load_image).batch(
        self.num_samples_per_class + 1)
    ways = shots.batch(self.num_classes)
    tasks = ways.batch(self.meta_batch_size)
    return tasks.map(lambda x: (tf.transpose(x, (0, 2, 1, 3, 4, 5)),
                                tf.eye(
                                    self.num_classes,
                                    batch_shape=(self.meta_batch_size, self.
                                                 num_samples_per_class + 1))))
