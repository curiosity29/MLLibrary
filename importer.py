
"""

 Create a dictionary "Vars" to load all the variable and library:
  for key, val in Vars:
    locals[key] = val

  with of type string


"""

import subprocess, sys, os, glob, importlib
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rasterio'])
# sys.path.append("../")


import subprocess, sys, os, glob
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rasterio'])


def findFolder(rootFolder, *names, sub_folder = False):
  folders = []
  for name_ in names:
    #glob.iglob
    folders_ = glob.glob(f"*{name_}*", root_dir = rootFolder, recursive=True)
    if len(folders_) >= 1:
      folders.append(
          os.path.join(rootFolder, folders_[0])
          )
    else:
      folders.append("")
  return folders


class Library():
  def __init__(self, version = 1):
    self.Vars = {}

  @staticmethod
  def get_modules(library_folder = "/content/drive/ColabShared/Library", installs = []):
    
    """
    Args:
      library_folder: path to the custom library to import from
      installs: array of name of modules to installs (e.g: ["rasterio"])
    Return dictionary of name: library

    """
    
    import subprocess, sys, os, glob
    for module_ in installs:
      subprocess.check_call([sys.executable, '-m', 'pip', 'install', module_])
    
    # lib = importlib.import_module
    import matplotlib as plt

    import rasterio as rs
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import keras
    from tensorflow.keras import layers, Model
    from sklearn.model_selection import train_test_split
    import tensorflow.keras.backend as K

    from datetime import datetime
    import io, itertools, sklearn.metrics

    import warnings
    warnings.filterwarnings("ignore")

    from functools import partial
    # from keras import ops

    import importlib


    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # from PIL import Image
    import os
    import cv2
    import numpy as nppp

    import json
    import imageio as imo
    # library = {
    #   "plt": lib("matplotlib.pyplot"),
    #   "tf": lib("tensorflow"),
    #   # "tensorflow.keras": "layers",
    #   # "tensorflow.keras": "Model"
    # }
    # !pip install tf-models-official
    # import tensorflow_models as tfm
    sys.path.append(library_folder)
    sys.path.append(os.path.join(library_folder, "Utility"))
    sys.path.append(os.path.join(library_folder, "Model"))

    # print(sys.path)
    from Utility import Inference, Preprocess, Dataset, Plot, Configs
    from Model import Blocks, Convolution, Losses

    from Plot.BasicPlot import plotI, plotM, plotPair, show2
    from Dataset.TFDataset import TFDataset
    from Dataset.TFRecords import create_records, file_iterate_auto
    from Preprocess import Augment, CropNResize, Normalize
    from Inference.InferHead import infer_head_max, infer_head_confident_level, softmaxHead
    from Inference import Window
    from Inference.Window import predict_windows

    from Convolution import UNet, DeeplabV3Plus, U2Net
    from Losses.Basnet import BasnetLoss
    from Callbacks import SaveWeights
    import Tensorboard.Plot
    from Losses.losses import Multi_stage_loss
    # from Loading import ImageLoader
    # from Convolution import *
    # importlib.reload(CropNResize)

    from Inference import Window
    from Convolution.Custom import uNet
    from Configs.Manual import get_args


    # import Convolution.Custom

    return locals()

  # @staticmethod
  def get_paths(self, work_folder):
    # work_folder = "/content/drive/MyDrive/ColabShared"
    library_folder = os.path.join(work_folder, "Library")

    work_folder = "/content/drive/MyDrive/ColabShared"
    library_folder = os.path.join(work_folder, "Library")

    util_folder, model_folder =\
    findFolder(library_folder, "Util", "Model")
    data_folder, checkpoint_folder =\
    findFolder(work_folder, "Training*/*18_1_v2", "Checkpoint*/Multi*segment*/Pools")

    setattr(self, "data_folder", data_folder)
    setattr(self, "util_folder", util_folder)
    # setattr(self, "data_folder", data_folder)
    # util_folder, model_folder, data_folder, checkpoint_folder

    def todict(**args):
      return args

    channels = 3
    n_class = 1
    SIZE = 512
    image_shape = (SIZE, SIZE, channels)
    label_shape = (SIZE, SIZE, n_class)
    args = todict(
        channels = channels,
        n_class = n_class,
        input_size = SIZE,
        output_size = SIZE
    )
    batch_size = 4

    setattr(self, "SIZE", SIZE)
    setattr(self, "image_shape", image_shape)
    setattr(self, "label_shape", label_shape)

    setattr(self, "batch_size", batch_size)

    return locals()
  # @staticmethod
  def get_dataset(self, **args):
    """
      additional args: batch_size and augmentation

    """
    batch_size = args["batch_size"]

    sys.path.append(self.util_folder)
    from Dataset.TFDataset import TFDataset
    from Dataset.TFRecords import create_records, file_iterate_auto
    from Preprocess import Augment
    import tensorflow as tf
    


    n_train = len(glob.glob(os.path.join(self.data_folder, "*train*")))
    n_test = len(glob.glob(os.path.join(self.data_folder, "*test*")))

    ## auto saving with name "train" and "test"
    # file_iterate_train, file_iterate_test =\
    #   file_iterate_auto("train"), file_iterate_auto("test")

    dataset_train = TFDataset(self.data_folder, n_train, "train", self.image_shape, self.label_shape)
    dataset_test = TFDataset(self.data_folder, n_test, "test", self.image_shape, self.label_shape)
    # n_train

    
    augment = Augment.Augment1(
    **args
    # probs = [0.5] * 5
    )

    batch_train = (
        dataset_train.cache()
        # .map(label_input)
        .shuffle(buffer_size= n_train)
        .repeat()
        .map(augment)
        # .map(aug)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    # augment = Augment.Augment1(center = False, rotate = None, size = (512,512))
    batch_test = (
        dataset_test#.cache()
        .shuffle(buffer_size= n_test)
        .repeat()
        # .map(label_input)
        # .map(augment)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return {
      "batch_train": batch_train,
      "batch_test": batch_test,
      "n_train": n_train,
      "n_test": n_test
    }

  
  @staticmethod
  def verbose_var(self, key, val):
    print(f"variable: {key} , with value: {val}")

  def verbose_lib(self, key, val):
    print(f"libarary: {key} , with name: {val}")

  def reload(self, module_name):
    """
      Return:
        reloaded module
    """
    return importlib.import_module(module_name)

  def load(self, variable_path):
    return

  def getVars(self, verbose = 0):
    if verbose == 1:
      for key, val in self.Vars.items():
        print(f"var: {key} , with value: {val}")
    return self.Vars
  