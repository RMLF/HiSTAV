from meshGIS.layerSHP import layerSHP
from meshGIS.layerTIFF import layerTIFF

a = 1
myLayerSHP = layerSHP("./test/Test.shp")
myLayerSHP.readSHPData()

myLayerTIFF = layerTIFF("./test/DTM_MATLAB.tif")
myLayerTIFF.readTIFFData()
a = 2
