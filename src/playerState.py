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
    def __hash__(self):#not complete
        return 0
    def breakIntoAttrs(self):
        res = []
        for item in self.inventory:
            res.append(PlayerState(inventory={item:self.inventory[item]}))
        for buff in self.buffs:
            res.append(PlayerState(buffs={buff:self.buffs[buff]}))
        res.append(PlayerState(lookedAt=self.lookedAt))
        return res
    def isPoolable(self):
        return self.lookedAt == None and len(self.buffs) == 0
    def clone(self):#not complete
        return PlayerState()
