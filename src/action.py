from gameObject import *
from playerState import *


class Action:
    def __init__(self,ps_res,ps_req,use_metric):
        self.ps_res = ps_res
        self.ps_req = ps_req
        self.use_metric = use_meteric
    def execute(self):
        do = 'nothing'
    def clone(self):
        return Action(self.ps_res,self.ps_req,self.use_metric)
    def __eq__(self,other):
        return self.ps_res == other.ps_res and self.ps_req == other.ps_req

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

class ActionFactory:

    #returns an action that fulfills the PlayerState requirement, or None if an action couldn't be produced
    def getActions(ps):
        return None
