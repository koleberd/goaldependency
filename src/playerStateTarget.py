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

    def addSolution(self,attrName,pss):
        self.attributeList[attrName].append(pss)
    def addParent(self,parent):
        self.parent = parent
    def isFulfilled(self):
        return self.satisfied >= self.ps
    def __hash__(self):
        flatItems = []
        for item in self.attributeList:
            flatItems.append(item)
            for sol in self.attributeList[item]:
                flatItems.append(id(sol))
        return hash((self.ps,self.satisfied,frozenset(flatItems)))#,self.parent,frozenset(flatItems)))
    def __eq__(self,other):
        return other != None and self.ps == other.ps
    def getCost(self,scalars,table):
        cheapest = None
        for attr in self.attributeList:
            for sol in self.attributeList[attr]:
                if cheapest == None or sol.getCost(scalars,table) < cheapest:
                    cheapest = sol.getCost(scalars,table)
        return cheapest
    def calculateCost(self,scalars,table):
        cheapest = None
        for attr in self.attributeList:
            for sol in self.attributeList[attr]:
                if cheapest == None or sol.calculateCost(scalars,table) < cheapest:
                    cheapest = sol.calculateCost(scalars,table)
        return cheapest
