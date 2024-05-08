from osgeo import gdal
from subprocess import call
import numpy as np
import os
import sys

def reproject(self, input_path, output_path, target_res):
        # data = gdal.Open(self.base_file, gdal.GA_ReadOnly)
        # geoTransform = data.GetGeoTransform()
        # prj = data.GetProjection()
        # resx = geoTransform[1]
        # resy = -geoTransform[5]

        # s=['gdal_translate', '-tr', '{}'.format(resx),'{}'.format(resy),'-of', 'GTiff', '-ot', 'Byte', '-a_nodata', '0','{}'.format(self.imagefile),'{}/image.tif'.format(self.fileprefix)]
        s=['gdalwarp', 
        '-tr', '{}'.format(target_res),'{}'.format(target_res),             ## resolution
        '-of', 'GTiff',                                         ## file type
        # '-ot', 'UInt16',
        # '-co', 'BIGTIFF=YES',                                        ## data type
        # # '-a_nodata', '0', 
        # "-t_srs", prj,                                     ## nodata map to
        input_path, output_path]           ## file path

        print(f"calling: {s}")
        call(s)
        # s1 = ["rio", 'cogeo', 'create', self.output_image_file_tmp, self.output_image_file]
        # call(s1)
        # print("done calling")
        # os.remove(self.output_image_file_tmp)
        # data=None

