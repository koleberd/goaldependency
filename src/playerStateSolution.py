from playerStateTarget import *
from actionTarget import *
from util import *
#once instnatiated, will produce one self.ps for every parent
#PSS is only supposed to satisfy one attribute of a PS, while a PST satisfies multiple attributes
class PlayerStateSolution:
    def __init__(self,ps):
        self.ps = ps #doesn't change, and represents the attribute that this PSS targets in its parent PST(s)
        self.children = [] #actionTargets
        self.parents = []  #PST
    def addChild(self,at):
        self.child.append(at)
    def addParent(self,pst):
        self.parents.append(pst)
    def removeChild(self,at):
        remove(self.children,at)
    def completeChild(self,at):#not complete
        #propagate child's returns upward
        removeChild(at)
    def removeParent(self,pst):
        remove(self.parents,pst)
    def getPS(self):
        return self.ps
    def getTotal(self):
        total = PlayerState()
        for child in self.children:
            total = total + child.getResult()
        return total
    def getRequired(self):
        total = self.ps.clone()#need to start with at least 1 stack so unaddable attributes tally up
        for parent in self.parents[1:]:
            total = total + self.ps
        return total
    def isFulfilled(self):
         return self.getTotal() > self.getRequired()
    def getExcess(self):
        if not self.isFulfilled():
            return PlayerState()
        return self.getTotal() - self.getRequired()
    def isTwin(self,other):#not complete
        return False
