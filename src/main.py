from action import *
from gameState import *
from playerState import *
from playerStateTarget import *
from playerStateSolution import *
from actionTarget import *
from playerStateFactory import *
from actionFactory import *
from playerStateTest import *
from graphviz import *




def getName(obj):
    if type(obj) == PlayerStateTarget:
        return 'PST - ' + (str(obj.ps) + ' - ' + str(id(obj))).replace(':',';')
    if type(obj) == PlayerStateSolution:
        return 'PSS - ' + (str(obj.ps) + ' - ' + str(id(obj))).replace(':',';')
    if type(obj) == ActionTarget:
        return ('AT - ' + str(obj.act.ps_res) + ' - ' + str(id(obj))).replace(':',';')
    return 'Not identifiable'

def graphTree(levelIndex,name):

    g = Digraph('Tree',filename=('trees/' + name + '.gv'),format='png')
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

    g.view()

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



def decomposePS(ps,name):
    proxyAT = ActionTarget(Action(ps,PlayerState(),0,None))
    actFactory = ActionFactory()

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

        if True:#skip pooling for now
            #pool each level
            for pss in levelIndex[-2]:
                for twinPss in levelIndex[-2]:

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

    printTree(levelIndex)
    #prune the tree
    #node pruned if nonzero requirements and zero children
    treeLevel = len(levelIndex)
    while(treeLevel >= 0):
        somethingRemoved = False
        treeLevel -= 1
        newLevelList = []
        for levelItem in levelIndex[treeLevel]:
            if treeLevel % 3 == 2:#AT level
                if levelItem.child == None and levelItem.getRequirement() != PlayerState():
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
                sols = 0
                for attr in levelItem.attributeList:
                    sols += len(levelItem.attributeList[attr])
                if sols == 0:
                    levelItem.parent.child = None
                    levelItem.parent = None
                    somethingRemoved = True
                else:
                    newLevelList.append(levelItem)

        if somethingRemoved:
            levelIndex[treeLevel] = newLevelList

    graphTree(levelIndex,name)
    nodecount = 0
    for level in levelIndex:
        nodecount += len(level)
    print('Name: ' + name)
    print('Levels: ' + str(len(levelIndex)-3))
    print('Nodes: ' + str(nodecount))
    print('Pools: ' + str(pools))
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



test()

#decomposePS(PlayerState(inventory={'stone':4}),'4stone_tree.png')
#decomposePS(PlayerState(inventory={'wood':10}),'10wood_tree.png')
#decomposePS(PlayerState(inventory={'wood':10,'stone':4}),'10wood_4stone_tree.png')
#decomposePS(PlayerState(inventory={'stone pickaxe':1}),'1stonepx_tree.png')
#decomposePS(PlayerState(inventory={'iron pickaxe':1}),'1ironpx_tree.png')
decomposePS(PlayerState(inventory={'wood':10}),'1stonepx_tree.png')
