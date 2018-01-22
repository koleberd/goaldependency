from action import *
from gameState import *
from playerState import *
from playerStateTarget import *
from playerStateSolution import *
from actionTarget import *
from playerStateFactory import *
from actionFactory import *
from graphviz import *
import viewer
from inventoryManager import *
import time
import pyautogui
from gameWorld2d import *
import gameController

def getName(obj):
    if type(obj) == PlayerStateTarget:
        return 'PST - ' + (str(obj.ps) + ' - ' + str(id(obj))).replace(':',';')
    if type(obj) == PlayerStateSolution:
        return 'PSS - ' + (str(obj.ps) + ' - ' + str(id(obj))).replace(':',';')
    if type(obj) == ActionTarget:
        return ('AT - ' + str(obj.act.ps_res) + ' - ' + str(id(obj))).replace(':',';')
    return 'Not identifiable'

#levelIndex in form of
#len % 3 == 0 -> PST
#len % 3 == 1 -> PSS
#len % 3 == 2 -> AT
def graphTree(levelIndex,name,selectedAT=None):
    print('Rendering graph for ' + name)
    g = Digraph('Tree',filename=('trees/' + name + '.gv'),format='png')
    for level in range(0,len(levelIndex)):
        for item in levelIndex[level]:
            if level % 3 == 0:
                g.attr('node',color='red')
                g.node(getName(item),label=('PST - ' + str(item) + ' - ' + str(item.tempCost)[:4]))
            if level % 3 == 1:
                if len(item.parents) > 1:
                    g.attr('node',style='filled')
                g.attr('node',color='blue')
                g.node(getName(item),label=('PSS - ' + str(item) + ' - ' + str(item.tempCost)[:4]))
                g.attr('node',style='unfilled')
            if level % 3 == 2:
                if item.child == None:
                    g.attr('node',style='filled')
                g.attr('node',color='green')
                if hash(item) == hash(selectedAT):
                    g.attr('node',color='purple')
                g.node(getName(item),label=('AT - ' + str(item) + ' - ' + str(item.tempCost)[:4]))
                g.attr('node',style='unfilled')

    for level in range(0,len(levelIndex)):
        for item in levelIndex[level]:
            if level % 3 == 0:
                for attr in item.attributeList:
                    for sol in item.attributeList[attr]:
                        #g.attr('node',fillcolor='blue:cyan')
                        #g.node(getName(item))
                        #g.attr('node',fillcolor='red:magenta')
                        #g.node(getName(sol))
                        g.edge(getName(item),getName(sol))

            if level % 3 == 1:
                for act in item.children:
                    g.edge(getName(item),getName(act))
            if level % 3 == 2:
                if item.child != None:
                    g.edge(getName(item),getName(item.child))
    g.render()

def printTree(levelIndex):
    print("=TREE=")
    for level in range(0,len(levelIndex)):
        print("LEVEL " + str(level))
        for item in levelIndex[level]:
            if level % 3 == 0:
                print(str(item.ps))
            if level % 3 == 1:
                print(str(item.ps))
            if level % 3 == 2:
                print(str(item.getRequirement()) + " = " + str(item.getResult()))
    print("=/TREE=")

def upwardPruneTree(levelIndex):
    #prune the tree
    #node pruned if nonzero requirements and zero children
    treeLevel = len(levelIndex)
    while(treeLevel >= 0):
        somethingRemoved = False
        treeLevel -= 1
        newLevelList = []
        for levelItem in levelIndex[treeLevel]:
            if treeLevel % 3 == 2:#AT level
                if levelItem.child == None and levelItem.getRequirement() != PlayerState() and levelItem.getRequirement() != None:
                    remove(levelItem.parent.children,levelItem)
                    levelItem.parent = None
                    somethingRemoved = True
                else:
                    newLevelList.append(levelItem)
            if treeLevel % 3 == 1:#PSS level
                if len(levelItem.children) == 0:
                    for parent in levelItem.parents:
                        remove(parent.attributeList[levelItem.ps],levelItem)
                    levelItem.parents = []
                    somethingRemoved = True
                else:
                    newLevelList.append(levelItem)
            if treeLevel % 3 == 0:#PST level
                #print('---')
                sols = True
                #print(len(levelItem.attributeList))
                for attr in levelItem.attributeList:
                    #print(attr)
                    #print(len(levelItem.attributeList[attr]))
                    sols &= len(levelItem.attributeList[attr]) != 0
                if not sols:
                    levelItem.parent.child = None
                    levelItem.parent = None
                    for attr in levelItem.attributeList:
                        for sol in levelItem.attributeList[attr]:
                            remove(sol.parents,levelItem)
                    somethingRemoved = True
                else:
                    newLevelList.append(levelItem)
        if somethingRemoved:
            levelIndex[treeLevel] = newLevelList

    #graphTree(levelIndex,name+'_prereprune')
    #printTree(levelIndex)

def downwardPruneTree(levelIndex):
    treeLevel = 0
    while(treeLevel < len(levelIndex)-1):
        somethingRemoved = False
        treeLevel += 1
        newLevelList = []
        for levelItem in levelIndex[treeLevel]:
            if treeLevel % 3 == 2:#AT level
                if levelItem.parent == None:

                    if levelItem.child != None:
                        levelItem.child.parent = None
                    somethingRemoved = True
                else:
                    newLevelList.append(levelItem)
            if treeLevel % 3 == 1:#PSS level
                if len(levelItem.parents) == 0:
                    for child in levelItem.children:
                        child.parent = None
                    somethingRemoved = True
                else:
                    newLevelList.append(levelItem)
            if treeLevel % 3 == 0:#PST level
                if levelItem.parent == None:
                    for attr in levelItem.attributeList:
                        for sol in levelItem.attributeList[attr]:
                            remove(sol.parents,levelItem)
                    somethingRemoved = True
                else:
                    newLevelList.append(levelItem)
        if somethingRemoved:
            levelIndex[treeLevel] = newLevelList

def decomposePS(ps,tree_name,actFactory):
    proxyAT = ActionTarget(Action(ps,PlayerState(),0,None))

    levelIndex = decomposeAT(proxyAT,actFactory)
    #levelIndex in form of
    #len % 3 == 0 -> PST
    #len % 3 == 1 -> PSS
    #len % 3 == 2 -> AT

    pools = 0
    while True:
        levelIndex.extend([[],[],[]])
        for leafAT in levelIndex[-4]:
            if leafAT.getRequirement() != PlayerState():
                layerInd = decomposeAT(leafAT,actFactory)
                levelIndex[-3].extend(layerInd[0])
                levelIndex[-2].extend(layerInd[1])
                levelIndex[-1].extend(layerInd[2])


        for i in range(0,len(levelIndex[-2])):
            for j in range(i,len(levelIndex[-2])):
                pss = levelIndex[-2][i]
                twinPss = levelIndex[-2][j]
                if (not pss is twinPss) and len(pss.parents) > 0 and len(twinPss.parents) > 0 and pss.isTwin(twinPss) and pss.getExcess().fulfills(twinPss.ps) and pss.ps.isPoolable():#pull off the reference to twinPss's last parent and give it to pss
                    pss.addParent(twinPss.parents[-1])
                    twinPss.parents[-1].attributeList[twinPss.ps].append(pss)
                    remove(twinPss.parents[-1].attributeList[twinPss.ps],twinPss)
                    del twinPss.parents[-1]
                    pools += 1

        for pss in levelIndex[-2]:#clean up levelIndex[-3] from pooling
            if len(pss.parents) == 0:
                for childAT in pss.children:
                    remove(levelIndex[-1],childAT)
        newPSSList = []
        for pss in levelIndex[-2]:#clean up levelIndex[-2] from pooling
            if len(pss.parents) != 0:
                newPSSList.append(pss)
        levelIndex[-2] = newPSSList

        if(len(levelIndex[-1]) == 0 and len(levelIndex[-2]) == 0 and len(levelIndex[-3]) == 0):#if no new AT's were created, remove the rows that werent' filled and break
            break


    del levelIndex[-3]
    del levelIndex[-2]
    del levelIndex[-1]

    upwardPruneTree(levelIndex)
    downwardPruneTree(levelIndex)

    '''
    leafcount = 0
    nodecount = 0

    for level in range(0,len(levelIndex)):
        nodecount += len(levelIndex[level])
        if level % 3 == 2:
            for item in levelIndex[level]:
                if item.child == None or item.getRequirement() == PlayerState():
                    leafcount += 1
    print('Name: ' + tree_name)
    print('Levels: ' + str(len(levelIndex)-3))
    print('Nodes: ' + str(nodecount))
    print('Leaf nodes: ' + str(leafcount))
    print('Branches eliminated through pooling: ' + str(pools))
    '''
    return levelIndex

def decomposeAT(at,factory):
    levels = [[],[],[]]
    prune = False
    for psatr in at.getRequirement().breakIntoAttrs():
        if at.isCyclicRequirement(psatr):
            prune = True
    if prune:
        return levels
    pst = PlayerStateTarget(at.getRequirement())
    levels[0].append(pst)
    at.addChild(pst)
    pst.addParent(at)

    for attr in pst.attributeList:
        for act in factory.getActions(attr):
            ps_req = act.ps_req
            at = ActionTarget(act)
            pss = PlayerStateSolution(attr)
            pss.addParent(pst)
            levels[1].append(pss)
            while not pss.isFulfilled():
                pss.addChild(at.clone())
            for pssAct in pss.children:
                pssAct.addParent(pss)
                levels[2].append(pssAct)
            pst.addSolution(attr,pss)

    return levels


def run2d(topPS,name,world):
    actFactory = ActionFactory('2D')
    levelIndex = decomposePS(topPS,name,actFactory)
    #graphTree(levelIndex,name + '_init')
    print('---- STARTING SIMUILATION  ----')
    steps = []
    times = {}
    invM = InventoryManager()
    w2d = world
    gs = GameState(ps=None,fov=None,inv=invM,flatworld=w2d)
    while(not levelIndex[0][0].isComplete()):
        scales = actFactory.scaleCosts(gs.fov)

        levelIndex[0][0].calculateCost(scales)
        #graphTree(levelIndex,name + '_' + str(step))
        selectedAT = levelIndex[0][0].select()
        if len(steps) == 0 or steps[-1] is not selectedAT:
            steps.append(selectedAT)
        exT = time.time()
        #graphTree(levelIndex,name + '_' + str(len(steps)),selectedAT)
        selectedAT.execute(gs)
        exT = time.time() - exT
        if selectedAT not in times.keys():
            times[selectedAT] = 0
        times[selectedAT] += exT
        #gs = viewer.getCurrentGameState(invM)
        downwardPruneTree(levelIndex)
        gs.cycle += 1

def run2d3d(config_name):
    with open(config_name) as jscf:
        config = json.load(jscf)

    #create dependency tree
    action_factory = ActionFactory()
    level_index = decomposePS(PlayerState.parsePlayerStateJSON(config['target_ps']),config['simulation_name'],action_factory)
    #graphTree(level_index,config['simulation_name'] + '_init')


    #set up inventory and world models
    inv_manager = InventoryManager(config['world_name_3d'])
    world_2d = GameWorld2d( config['world_2d_location'],
                            (config['2d_start'][0],config['2d_start'][1]),
                            (config['2d_end'][0],config['2d_end'][1]),
                            config['3d_bind'],config['2d_bind'],
                            (config['spawn_pos'][0],config['spawn_pos'][1]))


    print(world_2d.rayCast(45,20))



    return 0
    gs = GameState(ps=None,fov=None,inv=inv_manager,world_2d=world_2d,world_name_3d=config['world_name_3d'])

    #Issue inititialize world and sync player location
    print('---- STARTING SIMUILATION  ----')
    time.sleep(1)
    '''
    for cmd in config['world_init_3d']:
        gameController.executeCommand(cmd)
    '''
    gameController.syncLoc(gs)

    steps = [] #to keep track of the series of AT's as they're executed
    times = {} #to track the total time for each AT type

    while(not level_index[0][0].isComplete()):

        scales = action_factory.scaleCosts(gs.fov) #calculate cost scalars based on field of view
        level_index[0][0].calculateCost(scales) #apply cost scalars
        selected_at = level_index[0][0].select() #select at for execution
        if len(steps) == 0 or steps[-1] is not selected_at: #record selected AT
            steps.append(selected_at)
            graphTree(level_index,config['simulation_name'] + str(gs.world_step),selectedAT=selected_at)

        exT = time.time()
        selected_at.execute(gs) #execute AT
        exT = time.time() - exT
        if selected_at not in times.keys():
            times[selected_at] = 0
        times[selected_at] += exT

        #block if inventories don't match
        while inv_manager != InventoryManager(config['world_name_3d']).parseInventory():
            print("INVENTORY MISMATCH")
            pyautogui.keyDown('esc')
            pyautogui.keyUp('esc')
            time.sleep(.1)
            pyautogui.keyDown('esc')
            pyautogui.keyUp('esc')
            time.sleep(.1)

        downwardPruneTree(level_index) #prune tree to clean up in case an action completed
        gs.world_step += 1


run2d3d('json/simulation_configs/TEST_ENV4.json')
