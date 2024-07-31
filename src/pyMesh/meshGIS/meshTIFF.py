from meshGIS.layerTIFF import layerTIFF


class meshTIFF:

    def __init__(self):
        self.dtm = layerTIFF("./GIS/tif/dtm.tif")
        self.erosion = layerTIFF("./GIS/tif/erosion.tif")
        self.friction = layerTIFF("./GIS/tif/friction.tif")
        self.initial = layerTIFF("./GIS/tif/initial.tif")
