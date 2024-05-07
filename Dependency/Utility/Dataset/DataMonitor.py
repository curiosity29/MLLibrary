import numpy as np
import pandas as pd
import json
import os, glob
import csv
from ..Preprocess.Normalize import preprocess_info
import rasterio as rs


class DataMonitor():
    def __init__(self, monitor_file):
        self.monitor_file = monitor_file


    def get_info(self):
        with open(self.monitor_file) as src:
            info = json.load(src)
        return info
    def save_info(self, info):
        with open(self.monitor_file) as dest:
            json.dump(info, dest, indent = 4)
    
    def update(self, new_info):
        """ update the info file with a dict of (item_index -> info)"""
        info = self.get_info()
        info.update(new_info)
        self.save_info(info)

    def clean(self):
        df = pd.read_csv(self.monitor_file)
        # df_sorted = df.sort_values(by = ["item_index"], ascending=True)
        df = df.drop_duplicates(subset='item_index', keep='last')
        df.to_csv(self.monitor_file)

    def get_hard(self, num, metric_name):
        df = pd.read_csv(self.monitor_file)
        df = df.sort_values(by = [metric_name], ascending=False)
        return np.array(df["item_index"][:num])


class RawDataMonitor():
    def __init__(self, monitor_file, meta_file, input_image_folder, input_label_folder):
        self.monitor_file = monitor_file
        self.monitor_file_json = monitor_file.replace(".csv", ".json")
        self.meta_file = meta_file
        self.input_image_folder = input_image_folder
        self.input_label_folder = input_label_folder
        self.header = ("image_index", "lows", "highs", "means", "shape", "proportions", "background_proportion")

    def wipe(self):
        with open(self.monitor_file_json, "w") as dest:
            json.dump({}, dest, indent = 4)
        
        # with open(self.monitor_file, "w") as dest:
        #     writer = csv.writer(dest)
        #     writer.writerow(self.header)
            
            
    def append_image_data_stats(self, current_stats, image_index, image_path):
            """
                
            """
            # images_path = glob.glob(os.path.join(self.input_image_folder, "*.tif"))
                                    
                
                # dest.write(f"{item_index};{image_index};{corX};{corY}", delimiter = ";")
                # writer = csv.writer(dest)

            with rs.open(image_path) as src:
                meta = src.meta
                image = src.read()
            image = np.transpose(image, (1, 2, 0))
            lows, highs, means = preprocess_info(image)

            # writer.writerow(f"{item_index};{image_index};{corX};{corY}", delimiter = ";")
            # writer.writerow((image_index, lows, highs, means, image.shape))
            if str(image_index) not in current_stats.keys():
                current_stats[str(image_index)] = {}
            current_stats[str(image_index)].update(
                dict(
                    lows = list(lows),
                    highs = list(highs),
                    means = list(means),
                    shape = image.shape,
                )
            )

            return current_stats

    def append_label_data_stats(self, current_stats, image_index, label_path):
        with rs.open(label_path) as src:
            label = src.read()
        
        proportions = list(np.mean(label, axis = (1,2)))
        background_proportion = 1 - np.sum(proportions)
        if str(image_index) not in current_stats.keys():
            current_stats[str(image_index)] = {}
        current_stats[str(image_index)].update(
            dict(
                proportions = proportions,
                background_proportion = background_proportion,
            )
        )
        return current_stats

    def append_stats(self):
        if os.path.exists(self.monitor_file_json):
            with open(self.monitor_file_json) as src:
                stats = json.load(src)
        else:
            os.makedirs(os.path.split(self.monitor_file)[0], exist_ok=True)
            stats = {}

        images_path = glob.glob(os.path.join(self.input_image_folder, "*.tif"))
        labels_path = glob.glob(os.path.join(self.input_label_folder, "*.tif"))

        for image_path, label_path in zip(images_path, labels_path):
            
            index = image_index = int(image_path.split("_").pop().replace(".tif", ""))
            label_index = int(label_path.split("_").pop().replace(".tif", ""))
            assert image_index == label_index
            
            stats = self.append_image_data_stats(current_stats = stats, image_index = index, image_path = image_path)
            stats = self.append_label_data_stats(current_stats=stats, image_index = index, label_path=label_path)
            # index +=1
        # with open(self.monitor_file_json) as src:
        #     stats = json.load(src)
        
        # df = pd.DataFrame(data)
        # df.to_csv(self.monitor_file, index = False)
        with open(self.monitor_file_json, "w") as dest:
            json.dump(stats, dest, indent = 4)
            
        self.update_meta()
        return stats

    def update_meta(self):
        # df = pd.read_csv(self.monitor_file_json)
        with open(self.monitor_file_json) as src:
            list_stats = json.load(src)
        list_stats = list(list_stats.values())
        lows = list(np.mean([stats["lows"] for stats in list_stats], axis = 0))
        highs = list(np.mean([stats["highs"] for stats in list_stats], axis = 0))
        means = list(np.mean([stats["means"] for stats in list_stats], axis = 0))
        meta = {}
        meta["stats"] = {}
        meta["stats"].update(dict(
            lows = lows,
            highs = highs,
            means = means
        ))

        with open(self.meta_file, "w") as dest:
            json.dump(meta, dest, indent = 4)
        
    def execute(self, mode = "none"):
        match mode:
            case "none":
                return
            case "update":
                self.append_stats()
            case "wipe":
                self.wipe()
            case "reset":
                self.wipe()
                self.append_stats()
                # self.update_meta()
    



