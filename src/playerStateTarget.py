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
        self.temp_cost_up = 0
        self.temp_cost_down = 0
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
    '''
    def getCost(self,scalars,table={}):
        total = 0
        for attr in self.attributeList:
            cheapest = None
            for sol in self.attributeList[attr]:
                if cheapest == None or sol.getCost(scalars,table) < cheapest:
                    cheapest = sol.getCost(scalars,table)/len(sol.parents)
            total += cheapest
        self.temp_cost_up = total
        return total
    '''
    #calculates cost based on recursive calculateCost on children; no table lookups
    def calculateCostUp(self,scalars):
        total = 0
        for attr in self.attributeList:
            cheapest = None
            for sol in self.attributeList[attr]:
                if cheapest == None or sol.calculateCostUp(scalars) < cheapest:
                    cheapest = sol.calculateCostUp(scalars)/len(sol.parents)
            total += cheapest
        self.temp_cost_up = total
        return total
    def calculateCostDown(self,scalars):
        return self.calculateCostDownR(scalars,0)
    def calculateCostDownR(self,scalars,passed_cost):
        self.temp_cost_down = passed_cost
        for attr in self.attributeList:
            for sol in self.attributeList[attr]:
                if sol.temp_cost_down != 0 and passed_cost < sol.temp_cost_down:
                    sol.temp_cost_down = passed_cost
                for at_child in sol.children:
                    at_child.calculateCostDown(scalars,passed_cost)

    def calculateCostSetDown(self,passed_set):
        for attr in self.attributeList:
            for sol in self.attributeList[attr]:
                for at in sol.children:
                    at.calculateCostSetDown(passed_set)

    def calculateDepth(self,depth):
        for attr in self.attributeList:
            for sol in self.attributeList[attr]:
                for at in sol.children:
                    at.calculateDepth(depth)
    '''
    def select(self):
        #cAttr = None
        cSol = None
        for attr in self.attributeList:
            for sol in self.attributeList[attr]:
                if (cSol == None or sol.temp_cost_up < cSol.temp_cost_up) and (sol.ps.inFrontOf == None or len(self.attributeList) == 1):
                    cSol = sol
        return cSol.select()
    '''
    #adds ps to the attribute accumulation corresponding with pss.ps
    def updatePSS(self,pss,ps):
        self.attributeAccumulation[pss.ps] = ps + self.attributeAccumulation[pss.ps]
        #print(self.attributeAccumulation[pss.ps].fulfills(pss.ps))
        if self.attributeAccumulation[pss.ps].fulfills(pss.ps):
            for sol in self.attributeList[pss.ps]:
                sol.parents = []
            del self.attributeList[pss.ps]
        else:
            for sol in self.attributeList[pss.ps]:
                if sol is not pss:
                    sol.adjust(ps)
        if len(self.attributeList) == 0:
            if self.parent != None:
                self.parent.child = None
            self.parent = None
    def getLeafNodes(self):
        leafs = self.getLeafNodesR()
        clean_set = []
        for leaf in leafs:
            clean = True
            for cl in clean_set:
                if id(cl) == id(leaf):
                    clean = False
            if clean and ((leaf.getResult().inFrontOf != None and len(leaf.parent.parents[0].attributeList) == 1) or leaf.getResult().inFrontOf == None):#looked at can't be pooled so this is safe, but you would have to enforce priority order here
                clean_set.append(leaf)
        return clean_set
    def getLeafNodesR(self):
        leafs = []
        for attr in self.attributeList:
            for sol in self.attributeList[attr]:
                for at in sol.children:
                    if at.child == None or at.getRequirement() == PlayerState():
                        leafs.append(at)
                    else:
                        leafs.extend(at.child.getLeafNodesR())
        return leafs
