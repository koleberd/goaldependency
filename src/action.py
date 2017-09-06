import gameObject
import playerState
import gameState


class Action:
    def __init__(self):
        self.completed = False
    def execute(self,playerSt,gameSt):
        do = 'nothing'

class StaticAction(Action):
    def execute(self,playerSt,gameSt):
        self.completed = True

class VariableStaticAction(StaticAction):
    def __init__(self,obj):
        super().__init__(self)

class UserInputAction(VariableStaticAction):
    def __init__(self,obj):
        super().__init__(self)

class InteractWithGameObject(UserInputAction):
    def __init__(self,obj):
        super().__init__(self)
        self.obj = obj
    def execute(self,playerSt,gameSt):
        super().execute(self,playerSt,gameSt)
        if(self.obj.getName() == 'player inventory'):
            press = "e"
        else:
            press = "right click"
class PlaceObject(UserInputAction):
    def execute(self,playerSt,gameSt,object):
        super().execute(self,playerSt,gameSt)
        press = 'right click'
class HarvestResource(UserInputAction):
    def __init__(self,res):
        super().__init__(self)
        self.res = res
    def execute(self,playerSt,gameSt):
        super().execute(self,playerSt,gameSt)
        hold = 1 #1 second
        #do some calculation to calculate breaktime
        press = ('left click for %d seconds').format(hold)
class Move(UserInputAction):
    def __init__(self,direction='none',rotation='none',pitch='none'):
        super().__init__(self)
        self.direction = direction
        self.rotation = rotation
        self.pitch = pitch
    def execute(self,playerSt,gameSt):
        super().execute(self,playerSt,gameSt)
        hold = 1
        if 'in water' in playerSt.getStatusEffects():
            hold = 1.5
        switchDirection = {
            'forward'   : 'w',
            'left'      : 'a',
            'backward'  : 's',
            'right'     : 'd'
        }
        #do pitch and rotation calculations
        press ('press %s for %d seconds').format(switchDirection.get(self.direction,'nothing'),hold)

class ScriptAction(VariableStaticAction):
    def __init__(self,obj):
        super().__init__(self)
        self.obj = obj
class CraftObject(ScriptAction):
    def execute(self,playerSt,gameSt):
        super().execute(self,playerSt,gameSt)
class DropObject(ScriptAction):
    def execute(self,playerSt,gameSt):
        super().execute(self,playerSt,gameSt)
class UseObject(ScriptAction):
    def execute(self,playerSt,gameSt):
        super().execute(self,playerSt,gameSt)
class SwitchToObject(ScriptAction):
    def execute(self,playerSt,gameSt):
        super().execute(self,playerSt,gameSt)

class DynamicAction(Action):
    def execute(self,playerSt,gameSt):
        super().execute(self,playerSt,gameSt)

class ConditionalAction(DynamicAction):
    def isCompleted(playerSt,gameSt):
        return True
    def nextAction(playerSt,gameSt):
        return StaticAction()
    def execute(self,playerSt,gameSt):
        if isCompleted(playerSt,gameSt):
            completed = True
            return
        nextAction(playerSt,gameSt).execute(playerSt,gameSt)

class MoveToLocation(ConditionalAction):
    #target might be coordinates
    def __init__(self,target):
        self.target = target
    def isCompleted(playerSt,gameSt):
        return True
    def nextAction(playerSt,gameSt):
        return StaticAction()

class LocateObject(ConditionalAction):
    #target is a GameObject
    def __init__(self,target):
        super().__init__(self)
        self.target = target
    def isCompleted(playerSt,gameSt):
        return True
    def nextAction(playerSt,gameSt):
        return StaticAction()

class PickUpResource(ConditionalAction):
    #target is a Resource
    def __init__(self,target):
        super().__init__(self)
        self.target = target
    def isCompleted(playerSt,gameSt):
        return True
    def nextAction(playerSt,gameSt):
        return StaticAction()

class SequentialAction(DynamicAction):
    def __init__(self,children,actionList,yld):
        super().__init__()
        self.children = children
        self.childrenReference = children
        self.actionList = actionList
        self.actionListRefernce = actionList
        self.accumulation = {}
        self.currentAction = 0 #is incremented to match the action in actionList it's on
        self.yld = yld
    def isCompleted(self,playerSt,gameSt):
        compl = self.completed
        for child in self.children:
            if type(child) != Action:
                return False
            compl &= child.completed
        for act in actionList: #needs to be rewritten to account for actionList resetting of currentAction
            if type(act) != Action:
                return False
            compl &= act.completed
        return compl
    def reset(self):
        currentAction = 0
        self.actionList = self.actionListRefernce
    def execute(self,playerSt,gameSt):
        if self.isCompleted(playerSt,gameSt):
            completed = True
            return

        #if children aren't done, turn both children from strings into Actions if they aren't already
        #   pick a child and execute that child
        #else
        #   pick the bottommost action in action list
        #   turn it from a string into an action
        #   execute that action

        #return the action that was executed
