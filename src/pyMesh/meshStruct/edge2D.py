import math


class edge2D:

    def __init__(self):
        self.nodeIdx = []
        self.facetIdx = []
        self.id = 0
        self.physId = 0

    def getLength(self, allNodes):
        pointA = allNodes[self.nodeIdx[0]].coord
        pointB = allNodes[self.nodeIdx[1]].coord
        return pointA.distance(pointB)

    def getNormal(self, allNodes):
        return 0.0
