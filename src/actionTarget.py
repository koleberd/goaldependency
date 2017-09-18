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
    #clones without references
    def clone(self):
        return ActionTarget(self.act)
    def getResult(self):
        return self.act.ps_res
    def getRequirement(self):
        return self.act.ps_req
    def __eq__(self,other):
        return self.act == other.act and self.child == other.child and self.parent == other.parent
    def __ne__(self,other):
        return not (self == other)
    def execute(self):#not complete
        self.act.execute()
        #if it's complete, update parent pss with self.act.ps_res
    def isCyclicRequirement(self,ps):
        res = False
        if parent != None and len(parent.parents) > 0:
            res |= ps.fulfills(pst.attributeList[parent.ps])
            if parent.parents[0].parent != None:
                res |= parent.parents[0].parent.isCyclicRequirement(ps)
        return res
