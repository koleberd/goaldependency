from gameObject import *
from playerState import PlayerState

class Action:
    def __init__(self,ps_req,ps_res,cost,executionFunction):
        self.ps_res = ps_res
        self.ps_req = ps_req
        self.cost = cost
        self.executionFunction = executionFunction
    def execute(self):#not complete
        self.executionFunction()

    def __eq__(self,other):
        return self.ps_res == other.ps_res and self.ps_req == other.ps_req
    def __ne___(self,other):
        return not (self == other)
    def __hash__(self):#not complete
        return 0
