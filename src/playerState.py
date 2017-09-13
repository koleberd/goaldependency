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
    def __sub__(self,other):
        res = self.clone()
        for item in res.inventory:
            if other.inventory[item] != None:
                res.inventory[item] = res.inventory[item] - other.inventory[item]
        for buff in res.buffs:
            if other.buffs[buff] != None:
                res.buffs[buff] = res.buffs[buff] - other.buffs[buff]
        return res
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
    def isPoolable(self):
        return self.lookedAt == None and len(self.buffs) == 0
    def clone(self):
        return PlayerState()


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

def remove(ls,obj):
    for key, val in enumerate(ls):
        if val == obj:
            del ls[key]
            break


def decomposePS(ps, parentPSS, parentPST, parentAT):
    pst = PlayerStateTarget(ps)
    pst.addParent(parentAT)
    for attr in pst.attributeList:
        for act in ActionFactory.getActions(attr):
            ps_req = act.ps_req
            #check for pruning by checking parents' reqs with act's reqs
            prune = False
            pstParentPointer = parentPST
            while pstParentPointer != None:
                prune |= pstParentPointer.ps == ps_req # or if ps_req is satisfied by parent's excess.. parent PST or PSS?
                pstParentPointer = pstParentPointer.parent.parent[0].parent #may have to change if non-identical PSTs can pool into the same PSS
            if not prune:
                at = ActionTarget(act)
                pss = PlayerStateSolution(attr)
                pss.addParent(pst)
                while not pss.isFulfilled():
                    pss.addChild(at.clone())
                for pssAct in pss.children:
                    pssAct.addParent(pss)
                pst.addSolution(attr,pss)
                #pooling
                if pss.isPoolable():
                    for atRelative in parentPSS.children:
                        pstRelative = atRelative.child
                        for pssTwin in pstRelative.attributeList[attr]: #PST -> PSS is the forking point for decisions
                            if pssTwin.children[0] == pss.children[0] and not pssTwin is pss: #they're identical PSS's but not the same
                                if pss.getExcess() > attr:
                                    #replace the reference to pssTwin in pstRelative with pss, and change pssTwin and pss's parent lists accordingly
                                    remove(pstRelative.attributeList[attr],pssTwin)
                                    remove(pssTwin.parents,pstRelative)
                                    pstRelative.attributeList[attr].append(pss)
                                    pss.addParent(pstRelative)

                if ps_req != None:
                    for pssAct in pss:
                        pssAct.attachChild(decomposePS(ps_req,pss,pst,pssAct))
    return pst

class PlayerStateFactory:
    def getResourceByName():
        return None

#toplvlps = PlayerState()
#decomposePS()
