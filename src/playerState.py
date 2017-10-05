from gameObject import *


#PLEASE UNDERSTAND THAT MOST PLAYERSTATE METHODS ARE NOT COMMUTATIVE, INCLUDING ADDITION

class PlayerState:
    def __init__(self,inventory={},buffs={},lookedAt=None):
        self.inventory = inventory
        self.buffs = buffs
        self.lookedAt = lookedAt
    def parsePlayerStateJSON(obj):#not complete

        inv = obj['inventory'] if 'inventory' in obj.keys() else {}
        buf = obj['buffs'] if 'buff' in obj.keys() else {}
        lok = obj['lookedAt'] if 'lookedAt' in obj.keys() else None


        return PlayerState(inv,buf,lok)
    ## gets resource under cursor
    # return a GameObject
    def getObjectUnderCrosshairs(self):
        return self.lookedAt
    def getInventory(self):
        return self.inventory
    def getBuffs(self):
        return self.buffs
    def __eq__(self,other):
        if type(other) != type(self):
            return False
        res = True
        for item in self.inventory:
            if not item in other.inventory.keys() or self.inventory[item] != other.inventory[item]:
                res = False
        for buff in self.buffs:
            if not buff in other.buffs.keys() or self.buffs[buff] != other.buffs[buff]:
                res = False
        if self.lookedAt != other.lookedAt:
            res = False
        return res
    def __ne__(self,other):
        return not self.__eq__(other)
    def __add__(self,other):
        n_inv = self.inventory.copy()
        for item in other.inventory:
            if item in n_inv.keys():
                n_inv[item] += other.inventory[item]
            else:
                n_inv[item] = other.inventory[item]

        n_buf = self.buffs.copy()
        for buff in other.buffs:
            if buff in n_buf.keys():
                n_buf[buff] += other.buffs[buff]
            else:
                n_buf[buff] = other.buffs[buff]

        return PlayerState(n_inv,n_buf,self.lookedAt)
    def __sub__(self,other):
        n_inv = self.inventory.copy()
        for item in other.inventory:
            if item in n_inv.keys():
                n_inv[item] -= other.inventory[item]
            else:
                n_inv[item] = other.inventory[item] * -1
        nn_inv = {}
        for item in n_inv:
            if n_inv[item] != 0:
                nn_inv[item] = n_inv[item]
        n_buf = self.buffs.copy()
        for buff in other.buffs:
            if buff in n_buf.keys():
                n_buf[buff] -= other.buffs[buff]
            else:
                n_buf[buff] = other.buffs[buff] * -1
        nn_buf = {}
        for buff in n_buf:
            if n_buf[buff] != 0:
                nn_buf[buff] = n_buf[buff]
        return PlayerState(nn_inv,nn_buf,self.lookedAt)
    def __lt__(self,other):
        res = True
        for item in self.inventory:
            if other.inventory[item] != None and self.inventory[item] >= other.inventory[item]:
                res = False
        for buff in self.buffs:
            if other.buffs[buff] != None and self.buffs[buff] >= other.buffs[buff]:
                res = False
        return res
    def __le__(self,other):
        return not self.__gt__(other)
    def __gt__(self,other):
        res = True
        for item in self.inventory:
            if other.inventory[item] != None and self.inventory[item] <= other.inventory[item]:
                res = False
        for buff in self.buffs:
            if other.buffs[buff] != None and self.buffs[buff] <= other.buffs[buff]:
                res = False
        return res
    def __ge__(self,other):
        return not self.__lt__(other)
    def __hash__(self):
        return hash((frozenset(self.inventory.keys()),frozenset(self.inventory.items()),frozenset(self.buffs.keys()),frozenset(self.buffs.items()),self.lookedAt))
    def __str__(self):
        return '[' + str(self.inventory)[1:-1] + ' / ' + str(self.buffs)[1:-2] + ' / ' + str(self.lookedAt) + ']'
    def breakIntoAttrs(self):
        res = []
        for item in self.inventory:
            res.append(PlayerState(inventory={item:self.inventory[item]}))
        for buff in self.buffs:
            res.append(PlayerState(buffs={buff:self.buffs[buff]}))
        if self.lookedAt != None:
            res.append(PlayerState(lookedAt=self.lookedAt))
        return res
    def isPoolable(self):
        return self.lookedAt == None and len(self.buffs) == 0
    def isEmpty(self):
        return self.lookedAt == None and len(self.buffs) == 0 and len(self.inventory) == 0
    def isAttribute(self):
        res = len(self.buffs) + len(self.inventory)
        if self.lookedAt != None:
            res += 1
        return res == 1
    def fulfills(self,other):
        res = True
        for item in other.inventory:
            if not item in self.inventory.keys() or self.inventory[item] < other.inventory[item]:
                res = False
        for buff in other.buffs:
            if not buff in self.buffs.keys() or self.buffs[buff] < other.buffs[buff]:
                res = False
        res &= self.lookedAt == other.lookedAt
        return res
    def clone(self):#not complete
        return PlayerState()
    #return if the PS has the same attributes as self, but not necessarily at the same magnitudes
    def isParallel(self,other):
        res = True
        for key in other.inventory.keys():
            if not key in self.inventory.keys():
                res = False
        for key in other.buffs:
            if not key in self.buffs.keys():
                res = False
        res &= self.lookedAt == other.lookedAt
        return res
