import pyautogui
import time
from gameState import *
from playerState import *
from gameObject import *

TURN_TIME = .05
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
def craftObject(obj):
    print('crafting: ' + str(obj))
    return True
def invCraftObject(obj):
    print('inv crafting: ' + str(obj))
    return True
def harvestObject(obj,tool=None):
    print('harvesting: ' + str(obj))
    return True
def locateObject(obj,gs,alg=None):
    print('locating: ' + str(obj))
    if alg == None or alg == 'alg1':
        pathfind1(obj,gs)
    return gs.ps.lookedAt == obj


def pathfind1(obj,gs):
    pyautogui.moveRel(SCREEN_WIDTH/32,0,TURN_TIME)

    # return some lambda
def executeFunction(name,gs,params):
    #obj['function'].split(":")[0],obj['function'].split(":")[1]
    #name = obj.split(":")[0]
    #params = obj.split(":")[1].split(',')
    #print('--')
    #print(name)
    #print(params)
    if name == 'craftObject':
        return craftObject(params[0])
    if name == 'invCraftObject':
        return invCraftObject(params[0])
    if name == 'locateObject':
        if len(params) == 2:
            return locateObject(params[0],gs,params[1])
        else:
            return locateObject(params[0],gs)
    if name == 'harvestObject':
        if len(params) == 2:
            return harvestObject(params[0],params[1])
        else:
            return harvestObject(params[0])
    return False
