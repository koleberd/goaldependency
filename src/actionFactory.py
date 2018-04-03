from action import *
from playerState import *
import gameController
import os.path
import json


###
#parses json containing known actions
#produces actions that return a specified PS
###

class ActionFactory:

    def __init__(self,costs):
        with open('json/actionMemory.json') as jsfl:
            actMemory = json.load(jsfl)
        self.costs = costs
        self.actionMemory = []
        for key in actMemory:
            self.actionMemory.append(self.parseActionJSON(actMemory[key]))


    #parses json containing known actions
    def parseActionJSON(self,obj):
        psp = PlayerState.parsePlayerStateJSON(obj['prereq'])
        psr = PlayerState.parsePlayerStateJSON(obj['result'])
        cst = obj['cost']
        if obj['function'] in self.costs.keys():
            cst = self.costs[obj['function']]
        else:
            cst = 1
        func = lambda gs: gameController.executeFunction(obj['function'].split(":")[0],gs,obj['function'].split(":")[1].split(','))
        nm = obj['function']
        return Action(psp,psr,cst,func,nm)

    #produces actions that return a specified PS
    def getActions(self,ps):
        ret = []
        for act in self.actionMemory:
            if act.ps_res.isParallel(ps):
                ret.append(act)
        return ret

        '''
    def scaleCosts(self,scalars):
        res = {}
        if scalars == None:
            return res
        for act in self.actionMemory:
            ps = act.ps_res
            if ps.inFrontOf in scalars.keys():
                res[act] = scalars[ps.inFrontOf]
        return res
        '''
