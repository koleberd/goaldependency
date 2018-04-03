from playerState import PlayerState
###
#represents a standalone, executable action with a PS requirement and resultant, and a cost.
###

class Action:
    def __init__(self,ps_req,ps_res,cost,executionFunction,name=''):
        self.ps_res = ps_res
        self.ps_req = ps_req
        self.cost = cost
        self.executionFunction = executionFunction
        self.name = name
    #returns if completed
    def execute(self,gs):
        return self.executionFunction(gs)
    def __eq__(self,other):
        return other != None and self.ps_res == other.ps_res and self.ps_req == other.ps_req
    def __ne___(self,other):
        return not (self == other)
    def __hash__(self):#not complete
        return hash((self.ps_res,self.ps_req,self.cost))#,self.executionFunction))
