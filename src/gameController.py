import pyautogui
import time
import json
from gameState import *
from playerState import *
from gameObject import *
from inventoryManager import *

TURN_TIME = .05
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

with open('json/craftingIndex.json') as cfjs:
    craftingRecipes = json.load(cfjs)

def craftObject(obj,gs):
    print('crafting: ' + str(obj))
    if not obj in craftingRecipes.keys():
        return True
    try:
        pyautogui.rightClick()
        for item in craftingRecipes[obj]['inputs']:
            name = item.split(':')[0]
            col = item.split(':')[1].split(',')[0]
            row = item.split(':')[1].split(',')[1]
            iCoord = gs.inv.coordOf(name)
            cCoord = gs.inv.coordCbC(col,row)
            pyautogui.click(x=iCoord[0],y=iCoord[1])
            pyautogui.rightClick(x=cCoord[0],y=cCoord[1])
            pyautogui.click(x=iCoord[0],y=iCoord[1])
            gs.inv.withdraw(name,1)
        rcCoord = gs.inv.coordCbC(3,0)
        riCoord = gs.inv.coordNextEmpty()
        pyautogui.click(x=rcCoord[0],y=rcCoord[1])
        pyautogui.click(x=riCoord[0],y=riCoord[1])
        gs.inv.deposit(obj,craftingRecipes[obj]['output'])
        pyautogui.press('esc')
        return True
    except:
        #print('Could not complete script')
        raise Exception('Could not complete script')
        return False



def invCraftObject(obj,gs):
    print('inv crafting: ' + str(obj))
    if not obj in craftingRecipes.keys():
        return True

    pyautogui.press('e')
    for item in craftingRecipes[obj]['inputs']:
        name = item.split(':')[0]
        col = item.split(':')[1].split(',')[0]
        row = item.split(':')[1].split(',')[1]
        iCoord = gs.inv.coordOf(name)
        cCoord = gs.inv.coordInvC(col,row)
        pyautogui.click(x=iCoord[0],y=iCoord[1])
        pyautogui.rightClick(x=cCoord[0],y=cCoord[1])
        pyautogui.click(x=iCoord[0],y=iCoord[1])
        gs.inv.withdraw(name,1)
    rcCoord = gs.inv.coordInvC(3,0)
    riCoord = gs.inv.coordNextEmpty()
    pyautogui.click(x=rcCoord[0],y=rcCoord[1])
    pyautogui.click(x=riCoord[0],y=riCoord[1])
    gs.inv.deposit(obj,craftingRecipes[obj]['output'])
    pyautogui.press('esc')
    return True

def harvestObject(obj,gs,tool=None):
    print('harvesting: ' + str(obj))
    return True
def locateObject(obj,gs,alg=None):
    if gs.ps.lookedAt == obj and len(gs.fov) == 1:
        print('located: ' + str(obj))
        return True
    print('locating: ' + str(obj))
    if alg == None or alg == 'alg1':
        pathfind1(obj,gs)
    return False


def pathfind1(obj,gs):
    if gs.ps.lookedAt != obj:
        searchFor1(obj,gs)
    elif gs.ps.lookedAt == obj and len(gs.fov) != 1:
        moveTo1(obj,gs)

def moveTo1(obj,gs):
    pyautogui.keyDown('w')
    time.sleep(TURN_TIME)
    pyautogui.keyUp('w')

def searchFor1(obj,gs):
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
