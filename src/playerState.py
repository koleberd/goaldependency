from action import *
from gameObject import *

class PlayerState:
    def __init__(self,inventory={},buffs={},lookedAt=None):
        self.inventory = inventory
        self.buffs = buffs
        self.lookedAt = lookedAt
    ## gets resource under cursor
    # return a GameObject
    def getObjectUnderCrosshairs(self):
        return self.lookedAt
    def getInventory(self):
        return self.inventory
    def getBuffs(self):
        return self.buffs
    def __eq__(self,other):
        res = True
        for item in self.inventory:
            if self.inventory[item] != other.inventory[item]:
                res = False
        for buff in self.buffs:
            if self.buffs[buff] != other.buffs[buff]:
                res = False
        if self.lookedAt != other.lookedAt:
            res = False
        return res
    def __ne__(self,other):
        return not self.__eq__(other)
    def __add__(self,other):
        return PlayerState(self.inventory.append(other.inventory),self.buffs.append(other.buffs),self.lookedAt)
    def __lt__(self,other):
        res = True
        for item in self.inventory:
            if self.inventory[item] > other.inventory[item]:
                res = False
        for buff in self.buffs:
            if self.buffs[buff] > other.buffs[buff]:
                res = False
        return res
    def __le__(self,other):
        return not self.__gt__(other)
    def __gt__(self,other):
        res = True
        for item in self.inventory:
            if self.inventory[item] < other.inventory[item]:
                res = False
        for buff in self.buffs:
            if self.buffs[buff] < other.buffs[buff]:
                res = False
        return res
    def __ge__(self,other):
        return not self.__lt__(other)
    def breakIntoAttrs(self):
        res = []
        for item in self.inventory:
            res.append(PlayerState(inventory={item,self.inventory[item]}))
        for buff in self.buffs:
            res.append(PlayerState(buffs={buff,self.buffs[buff]}))
        res.append(PlayerState(lookedAt=self.lookedAt))
        return res
    def isPoolable():
        return lookedAt == None and len(buffs) == 0

class PlayerStateSolution:
    def __init__(self,ps_fulfilled):
        self.ps_fulfilled = ps_fulfilled
        self.actionTargets = []
    def addActionTarget(self,at):
        self.actionTargets.append(at)
    def isFulfilled(self):
        return False
    def getChildren(self):
        return self.actionTargets
    def getExcess(self):
        return None

class PlayerStateTarget:
    def __init__(self,ps_target):
        self.ps_target = ps_target

        self.psAttrList = {}
        attrs = self.ps_target.breakIntoAttrs()
        for ps in attrs:
            self.psAttrList[ps] = []

    def attributeList(self):
        return self.psAttrList.keys()
    def addSolution(self,attrName,pss):
        self.psAttrList[attrName].append(pss)

def decomposePS(ps, parentPSS, parentPST):
    pst = PlayerStateTarget(ps)
    for attr in pst.attributeList():
        for act in ActionFactory.getActions(ps)
            ps_req = act.ps_req
            #check for pruning by checking parents' reqs with act's reqs
            prune = False
            if not prune:
                at = ActionTarget(act)
                pss = PlayerStateSolution(attr)
                if ps_req != None:
                    at.attachChild(decomposePS(ps_req,pss,pst))
                while not pss.isFulfilled():
                    pss.addActionTarget(at.clone())
                pst.addSolution(pss)
                if pss.isPoolable():

                    do = 'some pooling operations'

    return pst

class PlayerStateFactory:
    def getResourceByName():
        return None

toplvlps = PlayerState()
decomposePS()
