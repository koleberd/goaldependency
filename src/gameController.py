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

def harvestObject(obj,gs,tool=None):#still needs to collect resource
    print('harvesting: ' + str(obj))
    #toolType = environmentIndex['obj']['toolType']
    toolLevel = 0
    if tool != None:
        toolLevel = toolIndex[tool]['type'][environmentIndex[obj]['toolType']]
        #swap to tool
        pyautogui.press('e')
        eqCoord = gs.inv.coordSlot(0,0)
        iCoord = gs.inv.coordOf(tool)
        pyautogui.click(x=iCoord[0],y=iCoord[1])
        pyautogui.click(x=eqCoord[0],y=eqCoord[1])
        pyautogui.click(x=iCoord[0],y=iCoord[1])
        pyautogui.press('esc')
        pyautogui.keyDown('1')
        pyautogui.keyUp('1')

        invc1 = gs.inv.invCoordOf(tool)
        hold = gs.inv.inventory[invc1[0]][invc1[1]]
        gs.inv.inventory[invc1[0]][invc1[1]] = gs.inv.inventory[0][0]
        gs.inv.inventory[0][0] = hold

    toolTime = environmentIndex[obj]['breakTime'][toolLevel]
    pyautogui.mouseDown()
    time.sleep(toolTime + .25)
    pyautogui.mouseUp()


    gs.inv.depositStack(obj,1)
    time.sleep(.4)
    pyautogui.keyDown('esc')
    pyautogui.keyUp('esc')
    time.sleep(.1)
    pyautogui.keyDown('esc')
    pyautogui.keyUp('esc')
    actualInv = InventoryManager().parseInventory()
    while gs.inv != actualInv:
        print('resource missing!')
        #print(gs.inv.inventory)
        #print(actualInv.inventory)

        pyautogui.keyDown('esc')
        pyautogui.keyUp('esc')
        time.sleep(.1)
        pyautogui.keyDown('esc')
        pyautogui.keyUp('esc')

        pyautogui.keyDown('w')
        time.sleep(.5)
        pyautogui.keyUp('w')
        actualInv = InventoryManager().parseInventory()
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
    elif gs.ps.lookedAt == obj and len(gs.fov) == 1:
        turnToward1(obj,gs)
    elif gs.ps.lookedAt == obj and len(gs.fov) != 1:
        moveTo1(obj,gs)


def moveTo1(obj,gs):
    return
    pyautogui.keyDown('w')
    time.sleep(TURN_TIME)
    pyautogui.keyUp('w')

def searchFor1(obj,gs):
    return
    pyautogui.moveRel(SCREEN_WIDTH/16,0,TURN_TIME)
    '''
    choice = random.randint(1,8)
    if choice > 4:
        pyautogui.moveRel(rightTurn*(choice%4),0,TURN_TIME)
    else:
        switch = ['w','a','s','d']
        pyautogui.keyDown(switch[choice-1])
        time.sleep(TURN_TIME)
        pyautogui.keyUp(switch[choice-1])
    '''

def turnToward1(obj,gs):
    return
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
