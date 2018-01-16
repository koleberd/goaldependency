import pyautogui
import time
import json
import math
import random
import numpy as np
from gameState import *
from playerState import *
from gameObject import *
from inventoryManager import *

TURN_TIME = .25
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
RIGHTTURN = 10



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
    if gs.pm.target['pos'][0] != 0: #not on eye level
        pyautogui.moveRel(None,450)
    pyautogui.mouseDown()
    time.sleep(toolTime + .25)
    pyautogui.mouseUp()
    if gs.pm.target['pos'][0] != 0: #not on eye level
        pyautogui.moveRel(None,-450)
    gs.world_2d.updateLoc(gs.pm.target['pos'],None)
    gs.pm.target = None



    gs.inv.depositStack(obj,1)
    time.sleep(.4)
    pyautogui.keyDown('esc')
    pyautogui.keyUp('esc')
    time.sleep(.1)
    pyautogui.keyDown('esc')
    pyautogui.keyUp('esc')
    actualInv = InventoryManager('TEST_ENV4').parseInventory()
    while gs.inv != actualInv:
        print('resource missing!')
        #print(gs.inv.inventory)
        #print(actualInv.inventory)

        pyautogui.keyDown('esc')
        pyautogui.keyUp('esc')
        time.sleep(.1)
        pyautogui.keyDown('esc')
        pyautogui.keyUp('esc')

        '''
        pyautogui.keyDown('w')
        time.sleep(.5)
        pyautogui.keyUp('w')
        '''

        actualInv = InventoryManager('TEST_ENV4').parseInventory()
    return True

def locateObject(obj,gs,alg=None):
    '''
    locate object is separated into 2 parts
    1) finding the object
    2) moving to the object
    If gs.pm.target == None, then the object hasn't been found
    otherwise, the object has been found and the avatar should be moving toward the target.

    '''
    if gs.pm.target == None or gs.pm.target['obj'] != obj: #has not located the object
        objs = gs.world_2d.findClosestLayered(obj,1)

        if len(objs) == 0:
            raise Exception("none of target object exist in this world")
        path = gs.world_2d.astar(gs.world_2d.pos, (objs[0][1],objs[0][2]))[:-1]
        gs.world_2d.saveWorld(path,gs.world_step)
        gs.pm.target = {'obj':obj,'path':path,'pos':objs[0]}
        print('new object',gs.pm.target)
    elif gs.pm.target['obj'] == obj:
        path = gs.pm.target['path']
        if len(gs.pm.target['path']) == 0:
            syncLoc(gs)
            gs.world_2d.yaw = 0
            print(gs.world_2d.pos,gs.pm.target['pos'],gs.world_2d.yaw)
            turnToward(gs,(gs.pm.target['pos'][1],gs.pm.target['pos'][2]))

            return True
        #curr_pos = gs.world_2d.pos
        next_pos = path[-1]
        turnToward(gs,next_pos)
        #print(gs.world_2d.pos,gs.world_2d.yaw,next_pos)
        moveForward(gs,1)
        #print(gs.world_2d.pos,gs.world_2d.yaw)
        gs.pm.target['path'] = path[:-1]


    return False

def executeFunction(name,gs,params):

    print(name + ': ' + str(params))
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



#MOVEMENT FUNCTIONS. REPONSIBLE FOR UPDATING gs.world_2d.pos, gs.world_2d.yaw
def turn(gs, angle):
    print(angle,int(angle/90))
    turns = int(angle/90)
    for t in range(0,abs(turns)):
        pyautogui.moveRel(np.sign(turns)*660,None)
        time.sleep(.1)

def turnToward(gs,pos):
    curr_pos = gs.world_2d.pos
    angle = 180 #top
    if curr_pos[0] < pos[0]:
        angle = 90 #right
    if curr_pos[0] > pos[0]:
        angle = -90 #left
    if curr_pos[1] < pos[1]:
        angle = 0 #bottom
    left_to_turn = angle - gs.world_2d.yaw
    turn(gs,-left_to_turn)
    gs.world_2d.yaw = angle


def moveForward(gs,units):
    #gs because speed boosts might be relevant
    pyautogui.keyDown('w')
    time.sleep(.132)
    pyautogui.keyUp('w')
    time.sleep(.15)

    gs.world_2d.pos = (gs.world_2d.pos[0]+int(math.sin(math.radians(gs.world_2d.yaw))),
           gs.world_2d.pos[1]+int(math.cos(math.radians(gs.world_2d.yaw))))



def executeCommand(cmd):
    pyautogui.press('t')
    pyautogui.typewrite(cmd,interval=.05)
    pyautogui.press('enter')
    time.sleep(.2)

def syncLoc(gs):
    cmd = 't/tp @s ' + str(gs.world_2d.offset_3d_x+gs.world_2d.pos[0]) + ' ' + str(gs.world_2d.elevation_3d) + ' ' + str(gs.world_2d.offset_3d_y+gs.world_2d.pos[1]) + ' 0 0'
    pyautogui.typewrite(cmd,interval=.05)
    pyautogui.press('enter')
    time.sleep(.211)



def bindWorlds(world_3d, world_2d, spawn_pos):
    spawn_3d_x = world_3d[0] - world_2d[0] + spawn_pos[0]
    spawn_3d_y = world_3d[2] - world_2d[1] + spawn_pos[1]

    cmd = 't/tp @s ' + str(spawn_3d_x) + ' ' + str(world_3d[1]) + ' ' + str(spawn_3d_y) + ' 0 0'
    pyautogui.typewrite(cmd,interval=.05)
    pyautogui.press('enter')
    time.sleep(.211)
