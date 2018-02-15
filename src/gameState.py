from playerState import *
from playerMemory import *
class GameState:
    def __init__(self,ps=PlayerState(),pm=PlayerMemory(),inv={},world_2d=None,world_step=0):
        self.ps = ps
        self.pm = pm
        self.inv = inv
        self.world_2d = world_2d
        self.world_step = world_step
