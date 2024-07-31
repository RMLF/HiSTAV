import math


class point2D:

    def __init__(self, inX=0.0, inY=0.0):
        self.x = inX
        self.y = inY

    def move(self, dx, dy):
        self.x = self.x + dx
        self.y = self.y + dy

    def distance(self, other):
        dx = self.x - other.X
        dy = self.y - other.Y
        return math.sqrt(dx**2 + dy**2)
