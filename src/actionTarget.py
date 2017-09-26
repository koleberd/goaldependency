from playerState import *
from action import *
from playerStateTarget import *
from playerStateSolution import *

class ActionTarget:
    def __init__(self,act):
        self.act = act
        self.child = None
        self.parent = None
    def getCost(self):
        return act.cost
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
        if self.parent != None and len(self.parent.parents) > 0:
            res |= ps.fulfills(self.parent.ps)
            if self.parent.parents[0].parent != None:
                res |= self.parent.parents[0].parent.isCyclicRequirement(ps)
        return res
    def __hash__(self):
        return hash((self.act,id(self.child),id(self.parent)))
    def __eq__(self,other):
        return other != None and self.act == other.act
    def calculateCost(self,scalars,table):
        scalar = 1
        if self.act in scalars.keys():
            scalar = scalars[self.act]
        res = self.getCost()*scalar
        if self.child != None and self.getRequirement() != PlayerState():
            res += child.calculateCost(scalars,table)
        return res
    def getCost(self,scalars,table):
        scalar = 1
        if self.act in scalars.keys():
            scalar = scalars[self.act]
        res = self.getCost()*scalar
        if self.child != None and self.getRequirement() != PlayerState():
            if not self.act in table.keys():
                table[self.act] = res + child.getCost(scalars,table)
            res = table[self.act]

        return res
