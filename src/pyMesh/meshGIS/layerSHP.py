import sys
import os
from osgeo import ogr
from meshStruct.geometry.point2D import point2D


class featureSHP:

    def __init__(self, inArrayPts=[], inArrayAttr=[]):
        self.points = inArrayPts
        self.attributes = inArrayAttr


class layerSHP:

    def __init__(self, inString=""):
        self.layerFile = inString
        self.type = ""
        self.features = []
        self.numFeatures = 0
        self.fields = []
        self.numFields = 0
        self.prjEPSG = 0

    def readSHPData(self, fileRequired=False):
        driverSHP = ogr.GetDriverByName("ESRI Shapefile")
        if driverSHP is None:
            sys.exit(1)
        else:
            if fileRequired and not os.path.exists(self.layerFile):
                sys.exit(1)
            elif os.path.exists(self.layerFile):
                dataSource = driverSHP.Open(self.layerFile, 0)
                if dataSource is None:
                    sys.exit(1)
                layer = dataSource.GetLayer()
                layerDefs = layer.GetLayerDefn()
                self.numFields = layerDefs.GetFieldCount()
                for field in range(self.numFields):
                    fieldName = layerDefs.GetFieldDefn(field).GetName()
                    self.fields.append(fieldName)
                self.numFeatures = layer.GetFeatureCount()
                for feature in layer:
                    featurePts = []
                    featureFlds = []
                    geometry = feature.GetGeometryRef()
                    for i in range(geometry.GetPointCount()):
                        pt = geometry.GetPoint(i)
                        featurePts.append(point2D(pt[0], pt[1]))
                    for i in range(self.numFields):
                        featureFlds.append(feature.GetField(i))
                    self.features.append(featureSHP(featurePts, featureFlds))
                self.type = feature.geometry().GetGeometryName()
                # message = "Found " + str(self.numFeatures) + " features of type " + geometryType
                # message = "CRS: EPSG" + str(self.prjEPSG)
                self.prjEPSG = layer.GetSpatialRef().GetAttrValue('AUTHORITY', 1)
                layer.ResetReading()
            else:
                return 0
                # message = "File not found, skipping"



# sys.stdout.write("Scanning SHP file: " + self.layerFile + " ...\n")
# sys.stderr.write("File " + os.path.abspath(self.layerFile) + " not present, can not proceed!\n")
# sys.stderr.write("Exiting now ...\n")