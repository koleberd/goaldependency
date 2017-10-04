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
        self.tempCost = 0
    def __hash__(self):#may need editing
        ids = []
        for child in self.children:
            ids.append(id(child))
        for parent in self.parents:
            ids.append(id(parent))
        return hash((self.ps,frozenset(ids)))
    def __eq__(self,other):
        return other != None and self.ps == other.ps
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
        res = self == other

        if 'wood pickaxe' in self.ps.inventory.keys() and not self is other and False:
            print('----')
            print(id(self))
            #print(self.parents[-1])
            #print(other.parents[-1])
            #print(self.parents[-1] == other.parents[-1])
            #print(self.parents[-1].parent)
            #print(other.parents[-1].parent)
            #print(self.parents[-1].parent == other.parents[-1].parent)
            #print(self.parents[-1].parent.parent)
            print(other.parents[-1].parent.parent)
        if len(self.parents) > 0 and len(other.parents) > 0 and self.parents[-1] == other.parents[-1] and self.parents[-1].parent != None and other.parents[-1].parent != None and self.parents[-1].parent == other.parents[-1].parent and self.parents[-1].parent.parent != None and other.parents[-1].parent.parent != None and self.parents[-1].parent.parent == other.parents[-1].parent.parent:
            if not self.parents[-1].parent.parent is other.parents[-1].parent.parent:
                res &= self.parents[-1].parent.parent.isTwin(other.parents[-1].parent.parent)
        else:
            res = False
        return res
    def getCost(self,scalars,table={}):
        total = self.children[0].calculateCost(scalars,table)
        if len(self.children) > 1:
            for child in self.children[1:]:
                print(self.ps)
                total += child.getCost(scalars,table)
        self.tempCost = total
        return total
    def calculateCost(self,scalars,table={}):
        total = 0
        for child in self.children:
            total += child.calculateCost(scalars,table)
        self.tempCost = total
        return total
    def select(self):
        return None
