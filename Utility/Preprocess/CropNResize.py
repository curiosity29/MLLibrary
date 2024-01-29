import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class Crop_and_resize(tf.keras.layers.Layer):
  def __init__(self, seed = 42, crop_size = (256, 256), out_size = (256,256)):
    super().__init__()
    self.seed = seed
    
    if crop_size is None:
      self.cropper = None
    else:
      self.cropper = layers.RandomCrop(crop_size[1], crop_size[0], seed = seed)

    if crop_size == out_size:
      self.resizer = None
    else:
      self.resizer = layers.Resizing(out_size[1], out_size[0])

  def call(self, inputs, labels):
    inputs, labels = tf.cast(inputs, tf.float32), tf.cast(labels, tf.float32)
    num_channels = inputs.shape[-1]
    stacked = tf.concat([inputs, labels], axis = -1)

    if self.cropper is not None:
      stacked = self.cropper(stacked)
    if self.resizer is not None:
      stacked = self.resizer(stacked)
    # output = self.resizer(rotated)

    # inputs = rotated[..., :num_channels]
    # labels = rotated[..., num_channels:]

    return stacked[..., :num_channels], tf.cast(stacked[..., num_channels:], dtype = tf.uint16)


def usefulCut(image, label, cutter = Crop_and_resize(), drop_threshold = 0.8, drop_unlabeled = True, **ignore):
  image, label = cutter(image, label)
  class_sum = np.sum(label, axis = (0,1))
  # contain no unlabeled:
  if drop_unlabeled and class_sum[0] > 0:
    return None
  # useful threshold
  if any( class_sum > np.sum(class_sum) * drop_threshold):
    return None

  if drop_unlabeled:
  # return label with removed unlabeled class
    return image, label[..., 1:]
  
  return image, label


def cutUntil(images, labels, usefulCut = usefulCut, size = (256, 256), n = 10, drop_threshold = 0.8, drop_unlabeled = True, **ignore):
  """
    drop if too high in 1 class
    could run indefinately
  """
  length = len(images)
  result_images, result_labels = [], []
  count = 0
  cutter = Crop_and_resize(crop_size = size, out_size = size)
  length = len(images)
  index = 0
  while count < n:
    # index = np.random.randint(0, length)
    cut = usefulCut(images[index], labels[index], cutter, drop_threshold = drop_threshold, drop_unlabeled = drop_unlabeled, **ignore)
    if cut is None:
      continue
    image_, label_ = cut
    result_images.append(image_)
    result_labels.append(label_)
    count = count + 1
    index = (index + 1) % length

  return result_images, result_labels


def usefulCut_binary(image, label, cutter = Crop_and_resize(), drop_threshold = 5, **ignore):
  """
  drop if low pixel count for 1
  """
  image, label = cutter(image, label)
  class_sum = np.sum(label, axis = (0,1))
  # contain no unlabeled:
  # if drop_unlabeled and class_sum[0] > 0:
  #   return None
  # print(class_sum[0])
  # useful threshold
  if any( class_sum < drop_threshold):
    return None
  
  return image, label