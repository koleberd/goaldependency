from playerState import *
from actionTarget import *
from playerStateSolution import *


class PlayerStateTarget:
    def __init__(self,ps):
        self.ps = ps
        self.satisfied = PlayerState()
        self.parent = None
        self.attributeList = {}
        attrs = self.ps.breakIntoAttrs()
        for ps in attrs:
            self.attributeList[ps] = []
        self.tempCost = 0
        self.attributeAccumulation = {} #keeps track of the total accumulated PS during runtime
        for ps in attrs:
            self.attributeAccumulation[ps] = PlayerState()
    def addSolution(self,attrName,pss):
        self.attributeList[attrName].append(pss)
    def addParent(self,parent):
        self.parent = parent
    def isFulfilled(self):
        return self.satisfied >= self.ps
    def isComplete(self):
        return len(self.attributeList) == 0
    def __hash__(self):
        flatItems = []
        for item in self.attributeList:
            flatItems.append(item)
            for sol in self.attributeList[item]:
                flatItems.append(id(sol))
        return hash((self.ps,self.satisfied,frozenset(flatItems)))#,self.parent,frozenset(flatItems)))
    def __eq__(self,other):
        return other != None and self.ps == other.ps
    def __str__(self):
        return str(self.ps)




    #-------------------------------------------
    #RUN TIME METHODS
    #-------------------------------------------

    #calculates cost based on recursive getCost calls on children and on table lookups if available
    def getCost(self,scalars,table={}):
        total = 0
        for attr in self.attributeList:
            cheapest = None
            for sol in self.attributeList[attr]:
                if cheapest == None or sol.getCost(scalars,table) < cheapest:
                    cheapest = sol.getCost(scalars,table)/len(sol.parents)
            total += cheapest
        self.tempCost = total
        return total
    #calculates cost based on recursive calculateCost on children; no table lookups
    def calculateCost(self,scalars,table={}):
        total = 0
        for attr in self.attributeList:
            cheapest = None
            for sol in self.attributeList[attr]:
                if cheapest == None or sol.calculateCost(scalars,table) < cheapest:
                    cheapest = sol.calculateCost(scalars,table)/len(sol.parents)
            total += cheapest
        self.tempCost = total
        return total


    def select(self):
        #cAttr = None
        cSol = None
        for attr in self.attributeList:
            for sol in self.attributeList[attr]:
                if cSol == None or ((sol.tempCost < cSol.tempCost and sol.ps.lookedAt == None) or (sol.ps.lookedAt != None and len(self.attributeList) == 1)):
                    cSol = sol
                    #cAttr = attr

        return cSol.select()
    #adds ps to the attribute accumulation corresponding with pss.ps
    def updatePSS(self,pss,ps):
        self.attributeAccumulation[pss.ps] = ps + self.attributeAccumulation[pss.ps]
        if self.attributeAccumulation[pss.ps].fulfills(pss.ps):
            del self.attributeList[pss.ps]
        else:
            for sol in self.attributeList[pss.ps]:
                if sol is not pss:
                    sol.adjust(ps)
        if len(self.attributeList) == 0:
            if self.parent != None:
                self.parent.child = None
            self.parent = None
