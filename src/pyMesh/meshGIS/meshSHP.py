from meshGIS.layerSHP import layerSHP


class meshSHP:

    def __init__(self):
        self.domain = layerSHP("./GIS/shp/domain.shp")
        self.lines = layerSHP("./GIS/shp/lines.shp")
        self.voids = layerSHP("./GIS/shp/voids.shp")
        self.boundaries = layerSHP("./GIS/shp/boundaries.shp")
        self.bndPoints = layerSHP("./GIS/shp/boundary_points.shp")
        self.targetMeshGenFolder = "./meshGen/"

    def readSHPFiles(self):
        self.domain.readSHPData(True)
        self.lines.readSHPData()
        self.voids.readSHPData()
        self.boundaries.readSHPData()
        self.bndPoints.readSHPData()

    def writeToGmsh(self):
        return 0

    def writeToTriangle(self):
        return 0