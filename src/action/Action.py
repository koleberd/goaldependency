class Action:
    def __init__(self):
    def execute(self):
    def parse():
        print 'done'




class StaticAction(Action):
class VariableStaticAction(StaticAction):
class InteractWithGameObject(VariableStaticAction):
class PlaceObject(InteractWithGameObject):#ignore
class HarvestResource(InteractWithGameObject):
class Move(VariableStaticAction):
    #contains information about key, key duration, and mouse movement.
    #namely, wasd, rotation, and pitch
class InventoryAction(VariableStaticAction):
class CraftObject(InventoryAction):
    def __init__(self,resources_with_positions):
        self.resources_with_positions = resources_with_positions
    def run(self):
        #find resources in inv, put them in crafing bench, note location of result
class DropObject(InventoryAction):#ignore
class UseObject(InventoryAction):#ignore
class SwitchToObject(InventoryAction):
    def __init__(self,target):
        self.target = target
    def execute(self):
        #do macro for switching to object

class DynamicAction(Action):
    def __init__(self,name,actionList,children):
        self.name = name
        self.actionList = actionList
        self.children = children
        self.completed = False

    def isCompleted(self):#wrong, needs to factor in actionList
        return True if self.completed
        self.completed = True
        for child in self.children:
            if not child.isCompleted():
                self.completed = False
        return self.completed



class LocateObject(DynamicAction):
class MoveToLocation(DynamicAction):
class ObtainResource(DynamicAction)
class AcquireResource(ObtainResource):
class AccessResource(ObtainResource):
class PickUpResource(DynamicAction):
class CompoundAction(DynamicAction):
