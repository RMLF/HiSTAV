import os
import sys
import numpy as np
from osgeo import gdal, osr
from meshStruct.geometry.point2D import point2D

gdal.UseExceptions()


class layerTIFF:

    def __init__(self, inString=""):
        self.layerFile = inString
        self.upperLeft = point2D
        self.cols = 0
        self.rows = 0
        self.dx = 0
        self.dy = 0
        self.data = np.zeros((1, 1))
        self.min = 0
        self.max = 0
        self.avg = 0
        self.dev = 0
        self.hasNoData = True
        self.prjEPSG = 0

    def readTIFFData(self):
        try:
            raster = gdal.Open(self.layerFile)
        except RuntimeError:
            sys.stderr.write("GDAL was unable to read " + os.path.abspath(self.layerFile) + "\n")
            sys.exit(1)
        transform = raster.GetGeoTransform()
        self.upperLeft = point2D(transform[0], transform[3])
        self.cols = raster.RasterXSize
        self.rows = raster.RasterYSize
        self.dx = transform[1]
        self.dy = transform[5]
        np.resize(self.data, (self.rows, self.cols))
        self.data = np.array(raster.GetRasterBand(1).ReadAsArray())
        stats = raster.GetRasterBand(1).ComputeStatistics(0)
        self.min = stats[0]
        self.max = stats[1]
        self.avg = stats[2]
        self.dev = stats[3]
        noDataVaue = raster.GetRasterBand(1).GetNoDataValue()
        self.hasNoData = np.any(self.data == noDataVaue)
        projection = osr.SpatialReference(wkt=raster.GetProjection())
        self.prjEPSG = projection.GetAttrValue('AUTHORITY', 1)
