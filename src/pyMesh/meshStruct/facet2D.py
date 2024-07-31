from meshStruct.geometry.point2D import point2D


class facet2D:

    def __init__(self):
        self.nodeIdx = []
        self.edgeIdx = []
        self.facetIdx = []
        self.center = point2D()
        self.id = 0
        self.physId = 0

    def setNodesCcWise(self):
        return 0.0

    def getArea(self):
        return 0.0

    def getSkewness(self):
        return 0.0

    def getMinEdge(self):
        return 0.0

    def getMaxEdge(self):
        return 0.0

    def getAspectRatio(self):
        return 0.0
