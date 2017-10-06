from playerState import *
from action import *
from playerStateTarget import *
from playerStateSolution import *

class ActionTarget:
    def __init__(self,act):
        self.act = act
        self.child = None
        self.parent = None
        self.tempCost = 0
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
    def __str__(self):
        return str(self.act.ps_res)
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


    #-------------------------------------------
    #RUN TIME METHODS
    #-------------------------------------------

    def calculateCost(self,scalars,table={}):
        scalar = 1
        if self.act in scalars.keys():
            scalar = scalars[self.act]
        res = self.act.cost*scalar
        if self.child != None and self.getRequirement() != PlayerState():
            res += self.child.calculateCost(scalars,table)
        self.tempCost = res
        return res
    def getCost(self,scalars,table={}):
        scalar = 1
        if self.act in scalars.keys():
            scalar = scalars[self.act]
        res = self.act.cost*scalar
        if self.child != None and self.getRequirement() != PlayerState():
            if not self.act in table.keys():
                table[self.act] = res + self.child.getCost(scalars,table)
            res = table[self.act]
        self.tempCost = res
        return res
    def select(self):
        if self.child == None:
            return self
        return self.child.select()
    def isComplete(self):
        return True
    def execute(self,gs):
        complete = self.act.execute(gs)
        if complete:
            self.parent.updateAT(self)
