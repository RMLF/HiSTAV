from meshStruct.geometry.point2D import point2D


class node2D:

    def __init__(self):
        self.coord = point2D()
        self.id = 0
        self.bed = 0
        self.frCoef = 0.0
        self.dzMax = 0.0
        self.grdId = 0
        self.physId = 0
