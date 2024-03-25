import rtree
import geopandas as gd

def polygons_f1(predict_path, label_path, threshold = None, return_count = True, return_error_count = True):
    
    gdf_predict = gd.read_file(predict_path)
    gdf_label = gd.read_file(label_path)
    if not gdf_predict.crs == gdf_label.crs:
        gdf_predict = gdf_predict.to_crs(gdf_label.crs)

    # get geometries from gdf
    label_geoms = list(gdf_label["geometry"])
    predict_geoms = list(gdf_predict["geometry"])


    # create rtree for label polygons 
    spatial_index = rtree.index.Index()
    for id, poly in enumerate(predict_geoms): 
        spatial_index.insert(id, poly.bounds)

    mean_accuracy = 0
    mean_recall = 0
    error_count = 0
    for id, label in enumerate(label_geoms):
        try:
            overlapping_indices = list(spatial_index.intersection(label.bounds))
        
            if len(overlapping_indices) == 0:
                continue
            if threshold is not None:
                best_IOU = threshold
                accuracy, recall = 0, 0
            else:
                best_IOU = -1.
            label_area = label.area
            for id_predict in overlapping_indices:
                predict = predict_geoms[id_predict]
                intersect = label.intersection(predict)
                
                predict_area = predict.area
                intersect_area = intersect.area
        
                IOU = intersect_area / (predict_area + label_area - intersect_area)
                if IOU  > best_IOU:
                    best_IOU = IOU
                    accuracy = intersect_area / label_area
                    recall = intersect_area / predict_area
                    
            mean_accuracy += accuracy
            mean_recall += recall
        except:
            error_count += 1
    count = len(label_geoms)
    mean_accuracy /= count
    mean_recall /= count
        
    
    # mean_accuracy, mean_recall
    result = dict(
        mean_accuracy = mean_accuracy, 
        mean_recall = mean_recall, 
    )
    if return_count:
        result["count"] = count
    if return_error_count:
        result["error_count"] = error_count

    return result

