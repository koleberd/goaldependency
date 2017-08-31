import json

#load GameObject index

#returns an array of parsed actions
def parseAction(name,actions):
    if 'note' in name:
        return []
    actionType = name.split(':')[0]
    actionName = [name.split(':')[1]]
    if '[' in actionName[0]:
        actionName = actionName[0].split('[')[1].split(']')[0].split(',')
    #for each action name, run create a new action set and use the index of the action name for any other isntances of []
    
    return actionName

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
    print(parseAction(key,actionIndexJson[key]))
