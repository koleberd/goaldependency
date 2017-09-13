from gameObject import *
from playerState import PlayerState
import json

class ActionFactory:
    def pathfind1(obj):
        do = 'nothing'
    def pathfind2(obj):
        do = 'nothing'
    def pathfind3(obj):
        do = 'nothing'
    pathfinders = {
        'alg1':pathfind1,
        'alg2':pathfind2,
        'alg3':pathfind3
    }
    def __init__(self):
        with open('json/resourceIndex.json') as jsfl:
            self.resourceIndex = json.load(jsfl)
    #returns an action that fulfills the PlayerState requirement, or None if an action couldn't be produced
    #@classmethod
    def getActions(self,ps):#not complete
        if not ps.isAttribute():
            return []
        ret = []
        if ps.lookedAt != None:#not tested
            for alg in pathfinders:
                ret.append(Action(PlayerState(lookedAt=ps.lookedAt),PlayerState(),0,pathfinders[alg](ps.lookedAt)))#read use metric from somewhere?
        if len(ps.inventory) == 1:

            resN,qnt = list(ps.inventory.items())[0]
            res = self.resourceIndex[resN]
            if res == None:
                return []
            if res['environmental'] != None:
                toolType = res['environmental']['toolType']
                toolLevel = res['environmental']['toolLevel']

                

        if len(ps.buffs) == 1:#not complete
            do = 'nothing'

        return ret
