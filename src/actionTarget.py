from playerState import *
from action import *
from playerStateTarget import *
from playerStateSolution import *

class ActionTarget:
    def __init__(self,act):
        self.act = act
        self.child = None
        self.parent = None
        self.temp_cost_up = 0
        self.temp_cost_down = 0
        #self.execMemory = {} #used by pathfinders etc to keep track of execution memory
    def addChild(self,child):
        self.child = child
    def addParent(self,parent):
        self.parent = parent
    #clones without references
    def clone(self):
        return ActionTarget(self.act)
    def getResult(self):
        return self.act.ps_res
    def getRequirement(self):
        return self.act.ps_req
    '''
    def __eq__(self,other):
        return self.act == other.act and self.child == other.child and self.parent == other.parent
    '''
    def __ne__(self,other):
        return not (self == other)
    def __str__(self):
        return str(self.act.ps_res)
    def isCyclicRequirement(self,ps):
        res = False
        if self.parent != None and len(self.parent.parents) > 0:
            res |= ps.fulfills(self.parent.ps)
            if self.parent.parents[0].parent != None:
                res |= self.parent.parents[0].parent.isCyclicRequirement(ps)
        return res
    def __hash__(self):
        return hash((self.act,id(self.child),id(self.parent)))

    def __eq__(self,other):
        return other != None and self.act == other.act


    #-------------------------------------------
    #RUN TIME METHODS
    #-------------------------------------------

    def calculateCostUp(self,scalars):
        scalar = 1
        if self.act in scalars.keys():
            scalar = scalars[self.act]
        res = self.act.cost*scalar
        if self.child != None and self.getRequirement() != PlayerState():
            res += self.child.calculateCostUp(scalars)
        self.temp_cost_up = res
        return res
    def calculateCostDown(self,scalars,passed_cost):
        scalar = 1
        if self.act in scalars.keys():
            scalar = scalars[self.act]
        self.temp_cost_down = passed_cost + self.act.cost*scalar
        if self.child != None and self.getRequirement() != PlayerState():
            self.child.calculateCostDownR(scalars,self.temp_cost_down)

    '''
    def getCost(self,scalars,table={}):
        scalar = 1
        if self.act in scalars.keys():
            scalar = scalars[self.act]
        res = self.act.cost*scalar
        if self.child != None and self.getRequirement() != PlayerState():
            if not self.act in table.keys():
                table[self.act] = res + self.child.getCost(scalars,table)
            res = table[self.act]
        self.temp_cost_up = res
        return res
    '''
    def select(self):
        if self.child == None:
            return self
        return self.child.select()
    def getDownwardCost(self):
        cheapest_parent = self.parent.parents[0]
        for ppt in self.parent.parents:
            do = 'nothing'
        self.act.cost
    def getNodeDepth(self):
        #print('---')
        #print(self.parent)
        #print(self.parent.parents[0])
        if self.parent.parents[0].parent != None:
            return self.parent.parents[0].parent.getNodeDepth() + 1
        else:
            return 0
    '''
    def isComplete(self):
        return True
    '''
    def execute(self,gs):
        '''
        execute(gs) is a pretty round-a-bout method so here's how it works
            1. call the action's <execute> method
            2. the action's execute method is a lambda which references a method in either gameController or gameController2d.
            3. This method parses the action's method string which looks something like 'harvestResource:wood,iron axe'
            4. Based on the method string, the parser method calls another method such as <harvestResource> in gameController(2d)
            5. This method actually executes keystrokes and manipulates the gamestate passed to it. this might mean manipulating the InventoryManager,
                checking to ensure the actual game inventory matches the inventoryManager (to ensure a block is picked up for example), or moving the
                position of the avatar in the 2d world (in the case of gameController2d)
        '''


        gs.pm.curr_at = self
        complete = self.act.execute(gs)
        if complete:
            #print("complete")
            self.parent.updateAT(self)

        return complete
