from playerState import *
class GameState:
    def __init__(self,ps=PlayerState(),fov={},inv={}):
        self.ps = ps
        self.fov = fov
        self.inv = inv
