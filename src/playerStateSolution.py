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
    def __hash__(self):#may need editing
        ids = []
        for child in self.children:
            ids.append(id(child))
        for parent in self.parents:
            ids.append(id(parent))
        return hash((self.ps,frozenset(ids)))
    def addChild(self,at):
        self.children.append(at)
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
            total = child.getResult() + total
        return total
    def getRequired(self):
        total = self.ps + PlayerState()#need to start with at least 1 stack so unaddable attributes tally up
        for parent in self.parents[1:]:
            total = self.ps + total
        return total
    def isFulfilled(self):
         return self.getTotal().fulfills(self.getRequired())
    def getExcess(self):
        if not self.isFulfilled():
            return PlayerState()
        return self.getTotal() - self.getRequired()
    def isTwin(self,other):#not complete
        res = True
        s_c = self
        o_c = other
        while s_c != None and o_c != None and res:
            res &= s_c == o_c
            if len(s_c.parents) > 0 and len(o_c.parents) > 0 and s_c.parents[0] == o_c.parents[0] and s_c.parents[0].parent != None and o_c.parents[0].parent != None and s_c.parents[0].parent == o_c.parents[0].parent and s_c.parents[0].parent.parent != None and o_c.parents[0].parent.parent != None:
                if s_cs_c.parents[0].parent.parent is o_c.parents[0].parent.parent:
                    break; #they are twins
                else:
                    s_c = s_cs_c.parents[0].parent.parent
                    o_c = o_c.parents[0].parent.parent
            else:
                res = False
                break;#kinda useless braek since loop will terminate anyway
        return res
