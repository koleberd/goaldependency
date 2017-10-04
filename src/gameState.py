from playerState import *
class GameState:
    def __init__(self,ps=PlayerState(),fov={}):
        self.ps = ps,
        self.fov = fov
