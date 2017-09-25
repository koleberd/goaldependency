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
        res = other != None and self.ps == other.ps
        return res
