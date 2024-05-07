from .WindowKernel import predict_windows_kernel
import glob, os
import numpy as np
from functools import partial
import rasterio as rs
from Dependency.Utility.Preprocess.Normalize import preprocess_info
from Dependency.Utility.Preprocess.Normalize import preprocess as image_preprocess
class Inferer():
    def __init__(self, image_folder, prediction_folder, predictor, preprocess,
        window_size = 512, input_dim = 3, predict_dim = 1, 
        output_type = "int8", batch_size = 1, step_divide = 4, 
        kernel = None
    ):
        self.image_folder = image_folder
        self.prediction_folder = prediction_folder
        self.last_index = 0

        self.predictor = predictor
        self.preprocess = preprocess
        self.window_size = window_size
        self.input_dim = input_dim
        self.predict_dim = predict_dim
        self.output_type = output_type
        self.batch_size = batch_size
        self.step_divide = step_divide
        self.kernel = kernel
        self.image_paths = np.array(glob.glob(os.path.join(self.image_folder, "*.tif")))

    def get_image_paths(self):
        self.image_paths = np.array(glob.glob(os.path.join(self.image_folder, "*.tif")))

    def infer(self, get_path = True, num = 1):
        if get_path:
            self.get_image_paths()
        images_path = self.image_paths
        if num == -1:
            num = len(images_path)

        ### get preprocess
        # if self.preprocess is None:
        ###
        
        for image_path in images_path[self.last_index: self.last_index + num]:
            image_name = os.path.basename(image_path)
            path_save = os.path.join(self.prediction_folder, image_name.replace(".tif", "_prediction.tif"))

            with rs.open(image_path) as src:
                meta = src.meta
                image = src.read()
                image = np.transpose(image, (1, 2, 0))
                lows, highs, means = preprocess_info(image)
                preprocess = partial(image_preprocess, lows = lows, highs = highs, map_to = (-1, 0 , 1))
            
            predict_windows_kernel(
                pathTif = image_path, 
                pathSave = path_save, 
                predictor = self.predictor, 
                preprocess = preprocess, 
                window_size = self.window_size, 
                input_dim = self.input_dim, 
                predict_dim = self.predict_dim, 
                output_type = self.output_type, 
                batch_size = self.batch_size, 
                step_divide = self.step_divide, 
                kernel = self.kernel
            )