from playerState import *
from action import *
from playerStateTarget import *
from playerStateSolution import *

class ActionTarget:
    def __init__(self,act):
        self.act = act
        self.child = None
        self.parent = None
    def getUseMeteric(self):
        return act.use_metric
    def addChild(self,child):
        self.child = child
    def addParent(self,parent):
        self.parent = parent
    def clone(self):
        res = ActionTarget(act.clone())
        if(childPST != None):
            res.attachChild(chlidPST.clone())
        return res
    def getResult(self):
        return self.act.ps_res
    def getRequirement(self):
        return self.act.ps_req
    def __eq__(self,other):
        return self.act.__eq__(other.act)
    def __ne__(self,other):
        return not self.__eq__(other)
