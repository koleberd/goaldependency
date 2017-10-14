from gameObject import *
from action import *
from playerState import *
import gameController

import json
class ActionFactory:

    def __init__(self):
        with open('json/resourceIndex.json') as jsfl:
            self.resourceIndex = json.load(jsfl)
        with open('json/actionMemory.json') as jsfl:
            actMemory = json.load(jsfl)
        self.actionMemory = []
        for key in actMemory:
            self.actionMemory.append(self.parseActionJSON(actMemory[key]))

    def parseActionJSON(self,obj):
        psp = PlayerState.parsePlayerStateJSON(obj['prereq'])
        psr = PlayerState.parsePlayerStateJSON(obj['result'])
        cst = obj['cost']
        func = lambda gs: gameController.executeFunction(obj['function'].split(":")[0],gs,obj['function'].split(":")[1].split(','))

        return Action(psp,psr,cst,func)

    #1. try to find an action with the desired output
    #2. if that can't be found, try one that hasn't been tried with the lowest cost??? - not sure if this is the right way to experiment
    def getActions(self,ps):#not complete
        ret = []
        for act in self.actionMemory:

            if act.ps_res.isParallel(ps):
                ret.append(act)
        return ret
    def scaleCosts(self,scalars):
        res = {}
        for act in self.actionMemory:
            ps = act.ps_res
            if ps.lookedAt in scalars.keys():
                res[act] = scalars[ps.lookedAt]
        return res
