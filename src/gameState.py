from playerState import *
from playerMemory import *

###
#contains a world, playerMemory, the player's inventory, and the current world time step
###



class GameState:
    def __init__(self,pm=PlayerMemory(),inv={},world_2d=None,world_step=0):
        self.pm = pm
        self.inv = inv
        self.world_2d = world_2d
        self.world_step = world_step
