import pyautogui
import time
import json
import random
from gameState import *
from playerState import *
from gameObject import *
from inventoryManager import *

TURN_TIME = .25
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
rightTurn = 600



with open('json/craftingIndex.json') as cfjs:
    craftingRecipes = json.load(cfjs)

with open('json/environmentIndex.json') as envjs:
    environmentIndex = json.load(envjs)

with open('json/toolIndex.json') as tooljs:
    toolIndex = json.load(tooljs)

def craftObject(obj,gs):
    print('crafting: ' + str(obj))
    if not obj in craftingRecipes.keys():
        return True
    try:
        for item in craftingRecipes[obj]['inputs']:
            name = item.split(':')[0]
            col = item.split(':')[1].split(',')[0]
            row = item.split(':')[1].split(',')[1]
            gs.inv.withdraw(name,1)
        gs.inv.deposit(obj,craftingRecipes[obj]['output'])
        return True
    except:
        #print('Could not complete script')
        raise Exception('Could not complete script')
        return False

def invCraftObject(obj,gs):
    print('inv crafting: ' + str(obj))
    if not obj in craftingRecipes.keys():
        return True
    for item in craftingRecipes[obj]['inputs']:
        name = item.split(':')[0]
        gs.inv.withdraw(name,1)
    gs.inv.deposit(obj,craftingRecipes[obj]['output'])
    return True

def harvestObject(obj,gs,tool=None):
    print('harvesting: ' + str(obj))
    #toolType = environmentIndex['obj']['toolType']
    toolLevel = 0
    if tool != None:
        toolLevel = toolIndex[tool]['type'][environmentIndex[obj]['toolType']]
        invc1 = gs.inv.invCoordOf(tool)
        hold = gs.inv.inventory[invc1[0]][invc1[1]]
        gs.inv.inventory[invc1[0]][invc1[1]] = gs.inv.inventory[0][0]
        gs.inv.inventory[0][0] = hold
    gs.inv.depositStack(obj,1)
    gs.flatworld.updateLoc(gs.flatworld.findClosest(obj,1)[0],None)

    return True

def locateObject(obj,gs,alg=None):
    print('locating: ' + str(obj))
    targetLoc = gs.flatworld.findClosest(obj,1)
    if len(targetLoc) != 0:
        path = gs.flatworld.astar(gs.flatworld.pos, targetLoc[0])
        gs.flatworld.pos = path[0]
        gs.flatworld.printWorld(path)
        
        return True
    return False


def executeFunction(name,gs,params):
    print(name)
    if name == 'craftObject':
        return craftObject(params[0],gs)
    if name == 'invCraftObject':
        return invCraftObject(params[0],gs)
    if name == 'locateObject':
        if len(params) == 2:
            return locateObject(params[0],gs,params[1])
        else:
            return locateObject(params[0],gs)
    if name == 'harvestObject':
        if len(params) == 2:
            return harvestObject(params[0],gs,params[1])
        else:
            return harvestObject(params[0],gs)
    return False
