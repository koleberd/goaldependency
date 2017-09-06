import json
import decomposer
from action import *
from gameState import *
from playerState import *
import gameObject
import playerState
import gameState

#returns the original string if not inclosed in brackets,
#   otherwise, returns the ind-th item in the brackets (separated with commas)
def bracketSplit(strg,ind):
    if type(strg) != str or '[' not in strg:
        return strg
    return strg.split('[')[0] + strg.split('[')[1].split(']')[0].split(',')[ind] + strg.split(']')[1]

#returns an array of parsed actions
def parseAction(name,actions):
    if 'note' in name:
        return {}
    actionType = name.split(':')[0]
    actionName = [name.split(':')[1]]

    actionSet = {}

    if '[' in actionName[0]:
        actionName = actionName[0].split('[')[1].split(']')[0].split(',')
        actionName = [el + name.split(']')[1] for el in actionName]
    #for each action name, run create a new action set and use the index of the action name for any other isntances of []
    for ind in range(0,len(actionName)):
        thisActionName = actionType + ':' + actionName[ind]
        actionSet[thisActionName] = {'actionList':[]}
        actionSet[thisActionName]['yields'] = actions['yields']
        actionSet[thisActionName]['children'] = {}

        for child in actions['children']:
            actionSet[thisActionName]['children'].update(
                {bracketSplit(child,ind):actions['children'][child]})
        for item in actions['actionList']:

            act,args = list(item.items())[0]
            actionSet[thisActionName]['actionList'].append({bracketSplit(act,ind):args})



    actionSetConverted = {}
    for name in actionSet:
        actionSetConverted.update(
            {name:SequentialAction( actionSet[name]['children'],
                                    actionSet[name]['actionList'],
                                    actionSet[name]['yields'])}
        )
    return actionSetConverted



def loadActionIndex():
    with open('json/actionIndex.json') as jsfl:
        actionIndexJson = json.load(jsfl)

    actionIndex = {}
    for key in actionIndexJson:
        actionIndex.update(parseAction(key,actionIndexJson[key]))

    for key in actionIndex:
        
        print(key + ':')
        for child in actionIndex[key].children:
            print(child + '-' + str(actionIndex[key].children[child]))
        print('then')
        for act in actionIndex[key].actionList:
            print(act)
        print()

    return actionIndex

def parseResource(name, tags):
    return {name, Resource(tags)}

def loadResourceIndex():
    resourceIndexJson = {}
    with open('json/resourceIndex.json') as jsfl:
        resourceIndexJson = json.load(jsfl)

    resourceIndex = {}
    for key in resourceIndexJson:
        resourceIndex.update({key:resourceIndexJson[key]})

    for key in resourceIndex:
        continue
        print('---')
        print(key)

    return resourceIndex

def parseActionFromName(name,args,index):
    switch = {
        "interactWithGameObject":InteractWithGameObject(name.split(':')[1]),
        "placeObject":PlaceObject(name.split(':')[1]),
        "harvestResource":HarvestResource(name.split(':')[1]),
        "move":Move(args),
        "craftObject":CraftObject(name.split(':')[1]),
        "dropObject":DropObject(args),
        "useObject":UseObject(args),
        "switchToObject":SwitchToObject(args),

        "moveToLocation":MoveToLocation(args),
        "locateObject":LocateObject(args),
        "pickUpResource":PickUpResource(args)
    }
    parsed = switch.get(name.split(':')[0])
    if parsed != None:
        return parsed
    else:
        return index(name)


def filterResources(tag):
    do = 'nothing'

def calculateAccumulations():
    do = 'nothing'
    #for every sequential action, calculate the cumulative yield of all of its children sequential actions with the same name and store them as a property of that sequential action

def selectAction(dtree,gameState,playerState,history):
    if type(dtree) == StaticAction or type(dtree) == ConditionalAction:
        return dtree
    #type must be SequentialAction
    return dtree

def decomposeAction(seqAction,index):
    for ind in range(0,len(seqAction.actionList)):
        name,args = list(seqAction.actionList[ind].items())[0]
        #print(name + '-' + str(args))

        seqAction.actionList[ind] = {name:parseActionFromName(name,args,index)}

    print(seqAction.actionList)





def loadAndRun():
    actionIndex = loadActionIndex()
    resourceIndex = loadResourceIndex()

    calculateAccumulations()

    goal = actionIndex['acquireResource:stick']

    #decomposeAction(goal,actionIndex)
    #print(goal.children)

    actionSelected = []
    actionsExecuted = [] # need to keep track of the action selected and the action executed

    gs = GameState()
    ps = PlayerState()

    while not goal.completed and False:
        #get visual input
        #process visual input (segment/cluster)

        #update gamestate and playerstate
        gs.update()
        ps.update()

        chosenAction = selectAction(goal,actionsExecuted)

        actionsSelected.append(chosenAction)
        actionsExecuted.append(chosenAction.execute(gs,ps))

loadAndRun()
