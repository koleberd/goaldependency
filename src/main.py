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


def decomposePS(ps, parentPSS, parentPST, parentAT):
    pst = PlayerStateTarget(ps)
    pst.addParent(parentAT)
    actFactory = ActionFactory()
    for attr in pst.attributeList:
        for act in actFactory.getActions(attr):
            ps_req = act.ps_req
            #check for pruning by checking parents' reqs with act's reqs
            prune = False
            pstParentPointer = parentPST
            while pstParentPointer != None:
                prune |= pstParentPointer.ps == ps_req # or if ps_req is satisfied by parent's excess.. parent PST or PSS?
                pstParentPointer = pstParentPointer.parent.parent[0].parent #may have to change if non-identical PSTs can pool into the same PSS
            if not prune:
                at = ActionTarget(act)
                pss = PlayerStateSolution(attr)
                pss.addParent(pst)
                while not pss.isFulfilled():
                    pss.addChild(at.clone())
                for pssAct in pss.children:
                    pssAct.addParent(pss)
                pst.addSolution(attr,pss)
                #pooling
                if pss.isPoolable():
                    for atRelative in parentPSS.children:
                        pstRelative = atRelative.child
                        for pssTwin in pstRelative.attributeList[attr]: #PST -> PSS is the forking point for decisions
                            if pssTwin.children[0] == pss.children[0] and not pssTwin is pss: #they're identical PSS's but not the same
                                if pss.getExcess() > attr:
                                    #replace the reference to pssTwin in pstRelative with pss, and change pssTwin and pss's parent lists accordingly
                                    remove(pstRelative.attributeList[attr],pssTwin)
                                    remove(pssTwin.parents,pstRelative)
                                    pstRelative.attributeList[attr].append(pss)
                                    pss.addParent(pstRelative)

                if ps_req != None:
                    for pssAct in pss:
                        pssAct.attachChild(decomposePS(ps_req,pss,pst,pssAct))
    return pst

def getName(obj):
    if type(obj) == PlayerStateTarget:
        return (str(obj.ps) + ' - ' + str(id(obj))).replace(':',';')
    if type(obj) == PlayerStateSolution:
        return (str(obj.ps) + ' - ' + str(id(obj))).replace(':',';')
    if type(obj) == ActionTarget:
        return (str(obj.act.ps_res) + ' - ' + str(id(obj))).replace(':',';')
    return 'Not identifiable'

def graphTree(levelIndex):

    g = Digraph('Tree',filename='trees/test_tree.gv',format='png')
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



def decomposePS2(ps):
    proxyAT = ActionTarget(Action(ps,PlayerState(),0,None))
    actFactory = ActionFactory()

    levelIndex = decomposeAT(proxyAT,actFactory)
    #levelIndex in form of
    #len % 3 == 0 -> PST
    #len % 3 == 1 -> PSS
    #len % 3 == 2 -> AT


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
                    if len(twinPss.parents) != 0: #if it hasn't alredy been pooled
                        if pss.isTwin(twinPss) and len(twinPss.parents) > 0 and pss.getExcess() > twinPss.ps:#pull off the reference to twinPss's last parent and give it to pss
                            pss.parents.append(twinPss.parents[-1])
                            remove(twinPss.parents[-1].attributeList[twinPss.ps],twinPss)
                            del twinPss.parents[-1]
                            pss.parents[-1].attributeList[pss.ps].append(pss)

            for pss in levelIndex[-2]:#clean up levelIndex[-3] from pooling
                if len(pss.parents) == 0:
                    for childAT in pss.children:
                        remove(levelIndex[-3],childAT)
            newPSSList = []
            for pss in levelIndex[-2]:#clean up levelIndex[-2] from pooling
                if len(pss.parents) != 0:
                    newPSSList.append(pss)
            levelIndex[-2] = newPSSList

        if(len(levelIndex[-1]) == 0 and len(levelIndex[-2]) == 0 and len(levelIndex[-3]) == 0):#if no new AT's were created, remove the rows that werent' filled and break

            break

    printTree(levelIndex)
    graphTree(levelIndex)


    #prune the tree
    #node pruned if nonzero requirements and zero children
    somethingRemoved = True
    treeLevel = len(levelIndex)
    while(somethingRemoved and treeLevel >= 0):
        somethingRemoved = False
        treeLevel -= 1
        newLevelList = []
        for levelItem in levelIndex[treeLevel]:
            if treeLevel % 3 == 2:
                if len(levelItem.children) == 0 and levelItem.getRequirement() != PlayerState():    #AT level
                    remove(levelItem.parent.children,levelItem)
                    levelItem.parent = None
                    somethingRemoved = True
                else:
                    newLevelList.append(levelItem)
            if treeLevel % 3 == 1:                                                                  #PSS level
                if len(levelItem.children) == 0:
                    for parent in levelItem.parents:
                        remove(parent.attributeList[levelItem.ps],levelItem)
                    levelItem.parents = []
                    somethingRemoved = True
                else:
                    newLevelList.append(levelItem)
            if treeLevel % 3 == 0:                                                                  #PST level
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




def decomposeAT(at,factory):
    levels = [[],[],[]]
    pst = PlayerStateTarget(at.getRequirement())
    levels[0].append(pst)
    at.addChild(pst)
    for attr in pst.attributeList:
        #print(str(attr))
        for act in factory.getActions(attr):
            #print("+")
            ps_req = act.ps_req
            prune = (ps_req.fulfills(attr) or at.isCyclicRequirement(ps_req)) and ps_req != PlayerState()
            #print(prune)
            if not prune:

                at = ActionTarget(act)
                pss = PlayerStateSolution(attr)
                pss.addParent(pst)
                levels[1].append(pss)
                #print(str(pss.ps))
                #print(str(pss.getTotal()))
                #print(str(pss.getRequired()))
                #print(pss.isFulfilled())
                #print(str(at.getResult()))
                while not pss.isFulfilled():
                    pss.addChild(at.clone())
                    #print('=')
                for pssAct in pss.children:
                    pssAct.addParent(pss)
                    levels[2].append(pssAct)
                pst.addSolution(attr,pss)
    return levels


#toplvlps = PlayerState()
#add things to the top level ps
#toplvlps.inventory['cobblestone'] = 200
#toplvlpst = decomposePS(toplvlps,None,None,None)


test()

tps = PlayerState(inventory={'wood':1})
tps2 = PlayerState(inventory={'wood':10})
#print(fact.actionMemory[0].ps_res.inventory)

#decomposePS2(tps2)
decomposePS2(PlayerState(inventory={'stone':2}))
