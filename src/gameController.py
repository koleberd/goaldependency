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
    if not obj in craftingRecipes.keys():
        return True

    for item in craftingRecipes[obj]['inputs']:
        name = item.split(':')[0]
        gs.inv.withdraw(name,1)
    gs.inv.deposit(obj,craftingRecipes[obj]['output'])
    return True

def invCraftObject(obj,gs):

    if not obj in craftingRecipes.keys():
        return True
    for item in craftingRecipes[obj]['inputs']:
        name = item.split(':')[0]
        gs.inv.withdraw(name,1)
    gs.inv.deposit(obj,craftingRecipes[obj]['output'])

    return True

def harvestObject(obj,gs,tool=None):#potentially will break if the player doesn't have a sufficient tool, but this shouldn't be reached
    #swap to correct tool
    toolLevel = 0
    if tool != None:
        toolLevel = 0
        #needs to withdraw one use from tool

    gs.world_2d.updateLoc(gs.pm.target['pos'],None)
    gs.pm.target = None
    gs.inv.depositStack(obj,1)

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
        objs = gs.world_2d.findClosest(obj,1)
        if len(objs) == 0:
            raise Exception("none of target object exist in this world: " + obj)
        path = gs.world_2d.astar(gs.world_2d.pos, (objs[0][0],objs[0][1]))[:-1]
        #gs.world_2d.saveWorld(path,gs.world_step)
        gs.pm.target = {'obj':obj,'path':path,'pos':objs[0]}
        #print('new object',gs.pm.target)
    elif gs.pm.target['obj'] == obj:
        path = gs.pm.target['path']
        if len(gs.pm.target['path']) == 0:
            #print(gs.world_2d.pos,gs.pm.target['pos'],gs.world_2d.yaw)
            turnToward(gs,(gs.pm.target['pos'][0],gs.pm.target['pos'][1]))

            return True
        #curr_pos = gs.world_2d.pos
        next_pos = path[-1]
        turnToward(gs,next_pos)
        #print(gs.world_2d.pos,gs.world_2d.yaw,next_pos)
        moveForward(gs,1)
        #print(gs.world_2d.pos,gs.world_2d.yaw)
        gs.pm.target['path'] = path[:-1]


    return False

def turnToward(gs,pos):#turns toward an adjacent position
    curr_pos = gs.world_2d.pos
    angle = 180 #top
    if curr_pos[0] < pos[0]:
        angle = 90 #right
    if curr_pos[0] > pos[0]:
        angle = -90 #left
    if curr_pos[1] < pos[1]:
        angle = 0 #bottom
    left_to_turn = angle - gs.world_2d.yaw

    gs.world_2d.yaw = angle

def moveForward(gs,units):#moves forward 1 unit
    gs.world_2d.pos = (gs.world_2d.pos[0]+int(math.sin(math.radians(gs.world_2d.yaw))),
           gs.world_2d.pos[1]+int(math.cos(math.radians(gs.world_2d.yaw))))

def executeFunction(name,gs,params):

    #print(name + ': ' + str(params))
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
