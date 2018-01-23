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
    gs.world_2d.saveWorld(gs.pm.target['full_path'],gs.world_step)
    gs.pm.target = None
    gs.pm.prev_at = None
    gs.pm.curr_at = None
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
    if obj != gs.pm.target['obj']:
        raise Exception("you went to the wrong place buddy")
    #swap to correct tool
    toolLevel = 0
    if tool != None:
        toolLevel = 0
        #needs to withdraw one use from tool

    gs.world_2d.updateLoc(gs.pm.target['pos'],None)
    gs.world_2d.saveWorld(gs.pm.target['full_path'],gs.world_step)
    gs.pm.target = None
    gs.pm.prev_at = None
    gs.pm.curr_at = None

    gs.inv.depositStack(obj,1)

    return True

def relink(at,pss,pst,pat):
    pss.addChild(at)
    at.addParent(pss)
    if len(pss.parents) == 0:
        if pss.ps not in pst.attributeList.keys():
            pst.attributeList[pss.ps] = []
        #print(str(pst))
        #print(str(pst.attributeAccumulation[pss.ps]))
        pst.attributeAccumulation[pss.ps] = PlayerState() - at.getResult() + pst.attributeAccumulation[pss.ps]
        #print(str(pst.attributeAccumulation[pss.ps]))
        pst.addSolution(pss.ps,pss)
        pss.addParent(pst)
        if pst.parent == None:
            #print(pss.parents)
            pat.addChild(pst)
            pst.addParent(pat)
            #print(str(pat))


def locateObject(obj,gs,alg=None):
    '''
    locate object is separated into 2 parts
    1) finding the object
    2) moving to the object
    If gs.pm.target == None, then the object hasn't been found
    otherwise, the object has been found and the avatar should be moving toward the target.
    '''
    if gs.pm.target == None or gs.pm.target['obj'] != obj or (gs.pm.target['obj'] == obj and gs.pm.prev_at != None and gs.pm.curr_at != None and id(gs.pm.prev_at) != id(gs.pm.curr_at)): #has not located the object or the locate_action has been completed but harvest was skipped
        if gs.pm.prev_at != None and gs.pm.prev_at.parent == None: #if the objects don't match because the previous one was completed but not harvested (harvest is trigger for gs.pm.target = None)
            relink(gs.pm.prev_at,gs.pm.prev_at_parent,gs.pm.prev_at_parent_parent,gs.pm.prev_at_parent_parent_parent)
        objs = gs.world_2d.findClosest(obj,1)
        if len(objs) == 0:
            raise Exception("none of target object exist in this world: " + obj)
        path = gs.world_2d.astar(gs.world_2d.pos, (objs[0][0],objs[0][1]))[:-1]
        #gs.world_2d.saveWorld(path,gs.world_step)
        gs.pm.target = {'obj':obj,'path':path,'pos':objs[0],'path_len':len(path),'full_path':path}
        #print('new object',gs.pm.target)
    elif gs.pm.target['obj'] == obj:
        path = gs.pm.target['path']
        if len(gs.pm.target['path']) == 0:
            gs.pm.metrics['distance traveled'] += gs.pm.target['path_len']

            #print(gs.world_2d.pos,gs.pm.target['pos'],gs.world_2d.yaw)

            #set things up in case of need for rewind
            gs.pm.prev_at = gs.pm.curr_at
            gs.pm.prev_at_parent = gs.pm.curr_at.parent
            gs.pm.prev_at_parent_parent = gs.pm.prev_at_parent.parents[0]
            gs.pm.prev_at_parent_parent_parent = gs.pm.prev_at_parent_parent.parent

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
