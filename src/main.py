from action import *
from gameState import *
from playerState import *
from playerStateTarget import *
from playerStateSolution import *
from actionTarget import *
from playerStateFactory import *
from actionFactory import *
from graphviz import *
from inventoryManager import *
import time
import random
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
    g = Digraph('Tree',filename=('simulation/trees/' + name + '.gv'),format='png')
    for level in range(0,len(levelIndex)):
        for item in levelIndex[level]:
            if level % 3 == 0:
                g.attr('node',color='red')
                g.node(getName(item),label=('PST - ' + str(item) + ' - ' + str(item.temp_cost_up)[:4] + ' - ' + str(item.temp_cost_down)[:4]))
            if level % 3 == 1:
                if len(item.parents) > 1:
                    g.attr('node',style='filled')
                g.attr('node',color='blue')
                g.node(getName(item),label=('PSS - ' + str(item) + ' - ' + str(item.temp_cost_up)[:4] + ' - ' + str(item.temp_cost_down)[:4]))
                g.attr('node',style='unfilled')
            if level % 3 == 2:
                if item.child == None:
                    g.attr('node',style='filled')
                g.attr('node',color='green')
                if hash(item) == hash(selectedAT):
                    g.attr('node',color='purple')
                g.node(getName(item),label=('AT - ' + str(item) + ' - ' + str(item.temp_cost_up)[:4] + ' - ' + str(item.temp_cost_down)[:4]))
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
    levelIndex[0][0].parent = None
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


# action target selection functions
def selectCheapest(at_arr):
    cheapest = at_arr[0]
    for at in at_arr:
        if at.temp_cost_up < cheapest.temp_cost_up:
            cheapest = at
    #print(cheapest.temp_cost_up)
    return cheapest
def selectMostExpensive(at_arr):

    exp = at_arr[0]
    for at in at_arr:
        if at.temp_cost_up > exp.temp_cost_up:
            exp = at
    return exp
def selectRandom(at_arr):
    return at_arr[random.randint(0,len(at_arr)-1)]
def selectSequential(at_arr):
    min_id = at_arr[0]
    for at in at_arr:
        if id(min_id) > id(at):
            min_id = at
    #print(id(min_id))
    return min_id

def selectDeepest(at_arr):
    depth = [0 for x in range(0,len(at_arr))]
    for i in range(0,len(at_arr)):
        depth[i] = at_arr[i].getNodeDepth()
    sel = 0
    for i in range(0,len(at_arr)):
        if depth[sel] < depth[i]:
            sel = i
    return at_arr[sel]
def selectMostShallow(at_arr):
    depth = [0 for x in range(0,len(at_arr))]
    for i in range(0,len(at_arr)):
        depth[i] = at_arr[i].getNodeDepth()
    sel = 0
    for i in range(0,len(at_arr)):
        if depth[sel] > depth[i]:
            sel = i
    return at_arr[sel]
def selectSmart(at_arr):
    #select a node with a highly reduced upward cost but a large downward cost
    return selectFirst(at_arr)



def run2d3d(config_name,select_method,select_name="",save_tree=False,save_path=False):
    full_start = time.time()
    with open(config_name) as jscf:
        config = json.load(jscf)

    #create dependency tree
    action_factory = ActionFactory()
    level_index = decomposePS(PlayerState.parsePlayerStateJSON(config['target_ps']),config['simulation_name'],action_factory)
    #graphTree(level_index,config['simulation_name'] + '_init')

    #set up inventory and world models
    inv_manager = InventoryManager()
    world_2d = GameWorld2d( config['world_2d_location'],(config['spawn_pos'][0],config['spawn_pos'][1]))

    gs = GameState(ps=None,fov=None,inv=inv_manager,world_2d=world_2d)

    print('---- STARTING SIMUILATION  ----')
    print('selection method: ' + str(select_name))

    steps = [] #to keep track of the series of AT's as they're executed


    full_start2 = time.time()
    root = level_index[0][0]
    while(not root.isComplete()):
        scales = action_factory.scaleCosts(gs.fov) #calculate cost scalars based on field of view
        root.calculateCostUp(scales) #apply cost scalars
        #root.calculateCostDown(scales)
        leaf_set = root.getLeafNodes()
        selected_at = select_method(leaf_set) #level_index[0][0].select() #select at for execution
        if len(steps) == 0 or id(steps[-1]) != id(selected_at): #record selected AT
            steps.append(selected_at)
            #print("------")
            #graphTree(level_index,config['simulation_name'] + '_' + str(gs.world_step),selectedAT=selected_at)


        selected_at.execute(gs) #execute AT


        #downwardPruneTree(level_index) #prune tree to clean up in case an action completed - only needed if the tree needs to be graphed
        #upwardPruneTree(level_index)
        gs.world_step += 1
    print(str(time.time()-full_start) + ' sec full run')
    print(str(time.time()-full_start2) + ' sec sim')
    print('metrics: ' + str(gs.pm.metrics))

#run2d3d('json/simulation_configs/TEST_ENV4.json')


#run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectMostShallow(x),select_name='most shallow')
run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectCheapest(x),select_name='cheapest')
#run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectSmart(x),select_name='smart')


'''
bound to loop
run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectSequential(x),select_name='sequential')
run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectRandom(x),select_name='random')
run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectMostExpensive(x),select_name='most expensive')
run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectDeepest(x),select_name='deepest')
'''
