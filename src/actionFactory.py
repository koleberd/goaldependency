from gameObject import *
from action import *
from playerState import *

import json

class ActionFactory:
    def pathfind1(obj):
        do = 'nothing'
    def pathfind2(obj):
        do = 'nothing'
    def pathfind3(obj):
        do = 'nothing'
        # return some lambda
    pathfinders = {
        'alg1':pathfind1,
        'alg2':pathfind2,
        'alg3':pathfind3
    }
    def __init__(self):
        with open('json/resourceIndex.json') as jsfl:
            self.resourceIndex = json.load(jsfl)
        with open('json/actionMemory.json') as jsfl:
            actMemory = json.load(jsfl)
        self.actionMemory = []

        for key in actMemory:

            self.actionMemory.append(self.parseActionJSON(actMemory[key]))


    #returns an action that fulfills the PlayerState requirement, or None if an action couldn't be produced
    #@classmethod
    def getFunction(self,name,args):
        return None
    def parseActionJSON(self,obj):


        psp = PlayerState.parsePlayerStateJSON(obj['prereq'])
        psr = PlayerState.parsePlayerStateJSON(obj['result'])
        cst = obj['cost']
        func = self.getFunction(obj['function'].split(":")[0],obj['function'].split(":")[1])

        return Action(psp,psr,cst,func)

    #1. try to find an action with the desired output
    #2. if that can't be found, try one that hasn't been tried with the lowest cost??? - not sure if this is the right way to experiment
    def getActions(self,ps):#not complete
        ret = []
        for act in self.actionMemory:

            if act.ps_res.isParallel(ps):
                ret.append(act)
        return ret
    def printCosts():
        print()
