import numpy as np
import rasterio as rs
# import tensorflow as tf
from rasterio.windows import Window as rWindow
from tqdm.auto import tqdm
import scipy
import os

# loader 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class WindowExtractor():
  def __init__(self, image_shape, window_shape, step_divide = 1):
    self.image_shape = image_shape
    self.window_shape = window_shape
    self.index = 0
    self.step_divide = step_divide
    self.stride_row = window_shape[0] // step_divide
    self.stride_col = window_shape[1] // step_divide
    self.n_row = (image_shape[0] - window_shape[0])  // self.stride_row + 2 # + 2 at start and finish
    self.n_col = (image_shape[1] - window_shape[0]) // self.stride_col + 2 # + 2 at start and finish
    self.row = 0
    self.col = 0
    self.total = self.getTotal()

  def __len__(self):
    return int(self.n_col * self.n_row)
  def getTotal(self):
    return int(self.n_col * self.n_row)

  def getRowCol(self, index):
    return index // self.n_col, index % self.n_col

  def next(self):
    self.row, self.col = self.getRowCol(self.index)
    self.index += 1
    if self.index > self.total:
      return (None, None), (None, None)
    return self.getWindow(self.row, self.col)

  def toRowCol(self, corX, corY):
    """
      get row and col index from pixel coordinate this does account for the last image in each row
    """
    row = corX // self.stride_row
    col = corY // self.stride_col
    # corX = row * self.window_shape[0] // self.step_divide
    # corY = col * self.window_shape[0] // self.step_divide
    return row, col
  def getByIndex(self, index):
      row, col = self.getRowCol(index)
      if self.index > self.total:
          return (None, None), (None, None)
      return self.getWindow(row, col)   
      
  def getWindow(self, row, col):
    """
    return top left coordinate and corner type: None, (0, 0), (0,1), ...
    -1: not corner
    0: first (left or top)
    1: last (right or bottem)
    """
    corner_type = [-1, -1]
    # print("col: ", col, self.n_col)
    # if col == self.n_col:
    #   print("none")
    #   return (None, None), (None, None)

    # print(row, col)
    # corX, corY = 0, 0
    # posY, posX = self.index // self.n_col, self.index % self.n_col
    if row == self.n_row-1:
      corner_type[1] = 1
      corX = self.image_shape[0] - self.window_shape[0]
    else:
      corX = row * self.stride_row

    if col == self.n_col-1:
      corner_type[0] = 1
      corY = self.image_shape[1] - self.window_shape[1]
    else:
      corY = col * self.stride_col

    if row == 0:
      corner_type[0] = 0
    if col == 0:
      corner_type[1] = 0

    return (corX, corY), corner_type

# windowExtractor = WindowExtractor(image_shape = (5000, 5000), window_shape = (512, 512), step_divide = 1)
# for _ in range(110):
#   window = windowExtractor.next()
#   print(window)
#   # print(window[0])
#   if window[0][0] is None:
#     break

def create_kernel(kernel, image_W, image_H,  window_size = 512, count = 1, step_divide = 1.25, dtype = "uint16", kernel_path = "./kernel.tif"):
    extractor = WindowExtractor(image_shape=(image_W, image_H), window_shape = (window_size, window_size), step_divide = step_divide)
    # create empty kernel
    with rs.open(kernel_path, "w", width = image_W, height = image_H, count = count, dtype = dtype) as dest:
        pass
        # while True:
        #     (corX, corY), corner_type = extractor.next()
        #     if corX is None:
        #         break
        #     window = rWindow(corX, corY, window_size, window_size)
        #     dest.write(np.zeros((1, window_size, window_size)), window = window)
    
    # create kernel weight
    extractor = WindowExtractor(image_shape=(image_W, image_H), window_shape = (window_size, window_size), step_divide = step_divide)
    
    with rs.open(kernel_path, "w+", width = image_W, height = image_H, count = count, dtype = dtype) as dest:
        
        while True:
            (corX, corY), corner_type = extractor.next()
            if corX is None:
                break
            window = rWindow(corX, corY, window_size, window_size)
            current = dest.read(window = window)[0]
            current += kernel.astype(dtype)
            dest.write(current[np.newaxis, ...], window = window)


def get_kernel(window_size):
    patch_weights = np.ones((window_size, window_size))
    patch_weights[0, :] = 0
    patch_weights[-1, :] = 0
    patch_weights[:, 0] = 0
    patch_weights[:, -1] = 0
    patch_weights = scipy.ndimage.distance_transform_edt(patch_weights) + 1
    # patch_weights = patch_weights[1:-1, 1:-1]
    kernel = patch_weights
    return kernel

def predict_windows_kernel(pathTif, pathSave, predictor, preprocess, window_size = 512, input_dim = 3, predict_dim = 1, 
output_type = "int8", batch_size = 1, step_divide = 4, kernel = None, temp_folder = "."):
    """
        combine predictions of a predictor in each window in a big tif image

        Args:
            pathTif: path of input tif image to predict
            pathSave: path of output tif image to save to
            predictor: prediction model that have input of a batch of image and output a batch of output with channel last
            preprocess: preprocess to apply to an image before using predictor
            window_size: size of image (shape of window_size x window_size x channel)
            input_dim: input image dimension/channel
            output_dim: output image dimension/channel
            output_type: output data type
            batch_size: size of each batch to use to predict
            step divide: size of stride to take window will be 1/step_divide
            kernel: kernel to combine predictions, set to None to use default

    """
    args = locals().copy()
    if kernel is None:
        kernel = get_kernel(window_size =window_size)

    args["kernel"] = kernel
    predict_windows(**args)
    kernel_path = os.path.join(temp_folder, "kernel.tif")
    
    with rs.open(pathSave) as src:
        meta = src.meta
        image_W = meta["width"]
        image_H = meta["height"]

    ## create big kernel 
    create_kernel(kernel, image_W, image_H,  window_size = window_size, count = 1, step_divide = step_divide, dtype = "uint16")
    
    # divide by kernel
                  
    extractor = WindowExtractor(image_shape=(image_W, image_H), window_shape = (window_size, window_size), step_divide = 1) # no need for overlapping
    with rs.open(pathSave, "r+") as dest:
        with rs.open(kernel_path, "r+") as big_kernel:
            
            while True:
                (corX, corY), corner_type = extractor.next()
                if corX is None:
                    break
                window = rWindow(corX, corY, window_size, window_size)
                kernel = big_kernel.read(window = window)
                pred = dest.read(window = window)
                pred = pred / kernel
                dest.write(pred, window = window)
                big_kernel.write(np.ones((1,window_size, window_size)), window = window)
            


class RasterDataset(Dataset):
    def __init__(self, image_path, window_size = 512, step_divide = 1, preprocess = None, dtype = "float32"):
        self.image_path = image_path
        self.window_size = window_size
        self.step_divide = step_divide
        self.reset(image_path = image_path, window_size=window_size, step_divide=step_divide)
        self.preprocess = preprocess
        self.dtype = dtype

    def reset(self, image_path, window_size = None, step_divide = None):
        if window_size is None:
            window_size = self.window_size
        if step_divide is None:
            step_divide = self.step_divide
        with rs.open(image_path) as src:
            meta = src.meta
        image_shape = (meta["width"], meta["height"])
        window_shape = (window_size, window_size)
        self.extractor = WindowExtractor(image_shape = image_shape, window_shape = window_shape, step_divide = step_divide)

    def __len__(self):
        return len(self.extractor)
    def __getitem__(self, idx):
        with rs.open(self.image_path, "r") as src:
            (corX, corY), corner_type = self.extractor.getByIndex(index = idx)
            # print(corX, corY, corner_type)
            window = rWindow(corX, corY, self.window_size, self.window_size)

            image = src.read(window = window)
            # image = np.transpose(image[:input_dim, ...], (1,2,0))

            if self.preprocess is not None:
                image = self.preprocess(image)

            return image.astype(self.dtype), (corX, corY), corner_type


def get_dataset(image_path, window_size = 512, step_divide = 1, preprocess = None):
    return RasterDataset(image_path = image_path, window_size = window_size, step_divide = step_divide, preprocess = preprocess)

def get_dataloader(dataset, batch_size = 4, num_workers = 1, shuffle = False, drop_last = False):
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, drop_last=False)


def predict_windows(pathTif, pathSave, predictor, preprocess, window_size = 512, input_dim = 3, predict_dim = 1, 
output_type = "int8", batch_size = 1, step_divide = 2, kernel = None, num_workers = 1, temp_folder= "."):
  """ temp folder currently not used """
#   kernel = kernel[..., np.newaxis]

  dataset = get_dataset(image_path = pathTif, window_size = window_size, step_divide = step_divide, preprocess= preprocess)
  dataloader = get_dataloader(dataset = dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False)
  with rs.open(pathTif) as src:
    # get meta
    out_meta = src.meta
    out_transform = src.transform
    profile = src.profile
    profile["transform"] = out_transform
    out_meta.update({"driver": "GTiff",
              "count": predict_dim, "dtype": output_type})

    image_W, image_H = out_meta["width"], out_meta["height"]
    extractor = WindowExtractor(image_shape=(image_W, image_H), window_shape = (window_size, window_size), step_divide = step_divide)

    with rs.open(pathSave, "w+", **out_meta) as dest:
      total = extractor.getTotal()
      total = total // batch_size if total % batch_size == 0 else total // batch_size + 1
      pogbar = tqdm(total = total) #, disable = True
      for index, batch in enumerate(dataloader):
        windows = []
        images, (corXs, corYs), corner_type = batch
        window = [rWindow(corX.numpy(), corY.numpy(), window_size, window_size) for corX, corY in zip(corXs, corYs)]
        # print("images :", images)
        # print("windows: ", windows)
        windows.append(window)
        
        predicts = predictor(images)

        for predict, window in zip(predicts, windows):
            ## channel first prediction
          predict = np.array([predict[channel_, ...] * kernel for channel_ in range(predict_dim)])
        #   predict = predict * kernel
        #   predict = np.transpose(predict, (2,0,1))


############ WAITING READ/ WRITE

          current = dest.read(window = window[0])
          predict = current + predict
          dest.write(predict, window = window[0])
          
############ WAITING READ/ WRITE

        pogbar.update(1)
      pogbar.close()




"""


Example usage:

pathTif = "./image.tif"
pathSave = "./prediction.tif"

def predict(batch):
    pred = model.predict(batch)
    pred = np.argmax(pred, axis = -1)
    pred = pred[..., np.newaxis]
    return pred        # shape: (batch, size, size, output_dim)

predictor = predict
preprocess = lambda x: x/255

predict_windows_kernel(pathTif, pathSave, predictor, preprocess, window_size = 512, input_dim = 3, predict_dim = 1, output_type = "int8", batch_size = 1, step_divide = 4, kernel = None)


"""











