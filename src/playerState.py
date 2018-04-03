#PLEASE UNDERSTAND THAT MOST PLAYERSTATE METHODS ARE NOT COMMUTATIVE, INCLUDING ADDITION

###
#Contains the PlayerState class. refer to the thesis document for more information about PlayerStates
###


class PlayerState:
    def __init__(self,inventory={},inFrontOf=None):
        self.inventory = inventory
        self.inFrontOf = inFrontOf
    def parsePlayerStateJSON(obj):#not complete

        inv = obj['inventory'] if 'inventory' in obj.keys() else {}
        lok = obj['inFrontOf'] if 'inFrontOf' in obj.keys() else None


        return PlayerState(inv,lok)
    ## gets resource under cursor
    # return a GameObject
    def getObjectUnderCrosshairs(self):
        return self.inFrontOf
    def getInventory(self):
        return self.inventory

    def __eq__(self,other):
        if type(other) != type(self):
            return False
        res = True
        for item in self.inventory:
            if not item in other.inventory.keys() or self.inventory[item] != other.inventory[item]:
                res = False
        if self.inFrontOf != other.inFrontOf:
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



        return PlayerState(n_inv,self.inFrontOf)
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

        return PlayerState(nn_inv,self.inFrontOf)
    def __hash__(self):
        return hash((frozenset(self.inventory.keys()),frozenset(self.inventory.items()),self.inFrontOf))
    def __str__(self):
        return '[' + str(self.inventory)[1:-1] + ' / ' + str(self.inFrontOf) + ']'
    def breakIntoAttrs(self):
        res = []
        for item in self.inventory:
            res.append(PlayerState(inventory={item:self.inventory[item]}))
        if self.inFrontOf != None:
            res.append(PlayerState(inFrontOf=self.inFrontOf))
        return res
    def isPoolable(self):
        return self.inFrontOf == None
    def isEmpty(self):
        return self.inFrontOf == None and len(self.inventory) == 0
    def isAttribute(self):
        res = len(self.inventory)
        if self.inFrontOf != None:
            res += 1
        return res == 1
    def fulfills(self,other):
        res = True
        for item in other.inventory:
            if not item in self.inventory.keys() or self.inventory[item] < other.inventory[item]:
                res = False
        res &= self.inFrontOf == other.inFrontOf
        return res
    def clone(self):#not complete
        return PlayerState()
    #return if the PS has the same attributes as self, but not necessarily at the same magnitudes
    def isParallel(self,other):
        res = True
        for key in other.inventory.keys():
            if not key in self.inventory.keys():
                res = False
        res &= self.inFrontOf == other.inFrontOf
        return res
