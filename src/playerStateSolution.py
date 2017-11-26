from playerStateTarget import *
from actionTarget import *
from util import *
#once instnatiated, will produce one self.ps for every parent
#PSS is only supposed to satisfy one attribute of a PS, while a PST satisfies multiple attributes


#pss.ps != pss.getTotal() initially. for example, the pss.ps for wood pickaxe = 1 but pss.getTotal() = 4 or however many units of pickaxe it returns
class PlayerStateSolution:
    def __init__(self,ps):
        self.ps = ps #doesn't change, and represents the attribute that this PSS targets in its parent PST(s)
        self.children = [] #actionTargets
        self.parents = []  #PST
        self.tempCost = 0
        self.adjustment = PlayerState()
    def __hash__(self):#may need editing
        ids = []
        for child in self.children:
            ids.append(id(child))
        for parent in self.parents:
            ids.append(id(parent))
        return hash((self.ps,frozenset(ids)))
    def __eq__(self,other):
        return other != None and self.ps == other.ps
    def __str__(self):
        return str(self.ps)
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
    #used for initilization, and projects the amount that will be produced.
    def getExcess(self):
        if not self.isFulfilled():
            return PlayerState()
        return self.getTotal() - self.getRequired()
    def isTwin(self,other):#not complete
        res = self == other


        if len(self.parents) > 0 and len(other.parents) > 0 and self.parents[-1] == other.parents[-1] and self.parents[-1].parent != None and other.parents[-1].parent != None and self.parents[-1].parent == other.parents[-1].parent and self.parents[-1].parent.parent != None and other.parents[-1].parent.parent != None and self.parents[-1].parent.parent == other.parents[-1].parent.parent:
            if not self.parents[-1].parent.parent is other.parents[-1].parent.parent:
                res &= self.parents[-1].parent.parent.isTwin(other.parents[-1].parent.parent)
        else:
            res = False
        return res

    #-------------------------------------------
    #RUN TIME METHODS
    #-------------------------------------------

    def getCost(self,scalars,table={}):
        total = self.children[0].calculateCost(scalars,table)
        if len(self.children) > 1:
            for child in self.children[1:]:
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
        return self.children[0].select()

    def isComplete(self):
        return len(self.children) == 0
    #removes the action target from children, adds its result to accumulation, and updates parents if necessary
    def updateAT(self,at):
        remove(self.children,at)
        at.parent = None
        accumulation = at.getResult()
        for parent in self.parents:
            if not accumulation.fulfills(self.ps): #case where multiple children and only one parent
                self.parents[-1].updatePSS(self,at.getResult())
                break
            if not parent.attributeAccumulation[self.ps].fulfills(self.ps):#case where multiple parents but only one child
                parent.updatePSS(self,self.ps)
                accumulation = accumulation - self.ps
        if self.isComplete():
            self.parents = []
    def adjust(self,ps):
        self.adjustment = self.adjustment + ps
        if self.adjustment.fulfills(self.children[-1].getResult()):
            self.adjustment = self.adjustment - self.children[-1].getResult()
            self.children[-1].parent = None
            del self.children[-1]
