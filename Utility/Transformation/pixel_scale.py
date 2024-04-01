import rasterio as rs
import numpy as np
def tif_scale(path_in, path_out, factor, zeros_padding = False):
    """
        scale down pixel size by a factor
    """
    with rs.open(path_in) as src:
        meta = src.meta
        transform = meta["transform"]
        transform = rs.Affine(transform[0] / factor,  transform[1],           transform[2],
                              transform[3],           transform[4] / factor,  transform[5]
                             )
                              
        # transform[0], transform[4] = transform[0] * factor, transform[4]* factor
        meta["transform"] = transform
        h, w = int(meta["height"] * factor), int(meta["width"] * factor)
        meta["height"], meta["width"]  = h, w
        image = src.read()
        zeros = np.zeros((image.shape[0], h, w))
        if zeros_padding:
            """ only work for odd integer factor """
            factor = int(factor)
            middle = factor//2
            zeros[:, middle::factor, middle::factor] = image
            image = zeros
            
        with rs.open(path_out, "w", **meta) as dest:
            dest.write(image[:, :h, :w])
"""
# Usage:       
path_in = "./Predictions/binary_boundary.tif"
path_out = "./test_binary_boundary.tif"
tif_scale(path_in, path_out, factor = 3, zeros_padding = True)

"""
