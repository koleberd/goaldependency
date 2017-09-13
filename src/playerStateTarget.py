from playerState import *
from actionTarget import *
from playerStateSolution import *

class PlayerStateTarget:
    def __init__(self,ps):
        self.ps = ps
        self.parent = None
        self.attributeList = {}
        attrs = self.ps.breakIntoAttrs()
        for ps in attrs:
            self.attributeList[ps] = []

    def addSolution(self,attrName,pss):
        self.attributeList[attrName].append(pss)
    def addParent(self,parent):
        self.parent = parent
