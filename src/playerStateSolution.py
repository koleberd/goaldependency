from playerStateTarget import *
from actionTarget import *

#once instnatiated, will produce one self.ps for every parent
#PSS is only supposed to satisfy one attribute of a PS, while a PST satisfies multiple attributes
class PlayerStateSolution:
    def __init__(self,ps):
        self.ps = ps
        self.children = [] #actionTargets
        self.parents = []  #PST
    def addChild(self,at):
        self.child.append(at)
    def addParent(self,parent):
        self.parents.append(parent)
    def getPS(self):
        return self.ps
    def getTotal(self):
        total = PlayerState()
        for child in children:
            total = total + child.getRequirement()
        return total
    def getRequired(self):
        total = self.ps.clone()#need to start with at least 1 stack so unaddable attributes tally up
        for parent in parents[1:]:
            total = total + self.ps
        return total
    def isFulfilled(self):
         return self.getTotal() > self.getRequired()
    def getExcess(self):
        if not self.isFulfilled():
            return PlayerState()
        return self.getTotal() - self.getRequired()
