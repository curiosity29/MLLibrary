from Utility.Inference.Window import WindowExtractor
from rasterio.windows import Window as rWindow
import rasterio as rs
import numpy as np

def tif_metrics(path_predict, path_label):
    with rs.open(path_predict) as predict_src:
        with rs.open(path_label) as label_src:
            meta = predict_src.meta
            window_size = 1000
            extractor = WindowExtractor(image_shape = (meta["width"], meta["height"]), 
                                        window_shape = (window_size, window_size), step_divide = 1)
            
            true_pos, true_neg, false_pos, fal_neg = 0, 0, 0, 0
            while True:
                (corX, corY), corner_type = extractor.next() 
                # if corner_type
                if corX is None:
                    break
                
                window = rWindow(corX, corY, window_size, window_size)
                predict = predict_src.read(window = window)
                label = label_src.read(window = window)

                true_pos = np.sum(predict * label, axis = (0,1,2))
                true_neg = np.sum((1-predict) * (1-label), axis = (0,1,2))
                false_pos = np.sum(predict * (1-label), axis = (0,1,2))
                false_neg = np.sum((1-predict) * label, axis = (0,1,2))

    sum = true_pos + true_neg + false_pos + false_neg
    # accuracy = (true_pos + true_neg)/ sum
    # accuracy = true_pos / (true_pos + false_pos)
    # recall = true_pos / (true_pos + false_neg)
    # f1 = 1 / (1/accuracy + 1/recall)
    # return accuracy, recall, f1

    return dict(
        true_pos = true_pos,
        true_neg = true_neg, 
        false_pos = false_pos, 
        false_neg = false_neg,
    )
            