import json

#load GameObject index

#returns the original string if not inclosed in brackets,
#   otherwise, returns the ind-th item in the brackets (separated with commas)
def bracketSplit(strg,ind):
    if type(strg) != str or '[' not in strg:
        return strg
    return strg.split('[')[0] + strg.split('[')[1].split(']')[0].split(',')[ind] + strg.split(']')[1]


#ind is used for bracketsplit
def parseRequirement(name,args,ind):
    if len(args) == 0:
        return {name:[]}
    return {(name + ':' + bracketSplit(args[0],ind)):args[1:]}

#returns an array of parsed actions
def parseAction(name,actions):
    if 'note' in name:
        return []
    actionType = name.split(':')[0]
    actionName = [name.split(':')[1]]

    actionSet = {}

    if '[' in actionName[0]:
        actionName = actionName[0].split('[')[1].split(']')[0].split(',')
        actionName = [el + name.split(']')[1] for el in actionName]
    #for each action name, run create a new action set and use the index of the action name for any other isntances of []
    for ind in range(0,len(actionName)):
        thisActionName = actionType + ':' + actionName[ind]
        actionSet[thisActionName] = {'actionList':{}}
        for key in actions:
            if 'yield' in key:
                actionSet[thisActionName]['yield'] = actions[key]
            elif key == 'children':
                actionSet[thisActionName]['children'] = {}
                print (actions['children'])
                for req in actions['children']:
                    actionSet[thisActionName]['children'].update({bracketSplit(req,ind):actions['children'][req]})
            else:
                actionSet[thisActionName]['actionList'].update({bracketSplit(key,ind):actions[key]})



    return actionSet

with open('json/resourceIndex.json') as jsfl:
    resourceIndexJson = json.load(jsfl)

resourceIndex = {}

for key in resourceIndexJson:
    load = 'nothing'

#load action index
with open('json/actionIndex.json') as jsfl:
    actionIndexJson = json.load(jsfl)

actionIndex = {}

for key in actionIndexJson:
    actionIndex.update(parseAction(key,actionIndexJson[key]))

for key in actionIndex:
    print(key)
    print(actionIndex[key])
