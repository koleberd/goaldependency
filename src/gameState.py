from playerState import *
from playerMemory import *
class GameState:
    def __init__(self,ps=PlayerState(),pm=PlayerMemory(),fov={},inv={},world_2d=None,world_step=0,world_name_3d=""):
        self.ps = ps
        self.pm = pm
        self.fov = fov
        self.inv = inv
        self.world_2d = world_2d
        self.world_step = world_step
        self.world_name_3d = world_name_3d
