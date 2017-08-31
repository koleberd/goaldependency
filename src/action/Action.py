class Action:
    def parse():
        print 'done'







class StaticAction(Action):
class VariableStaticAction(StaticAction):
class InteractWithGameObject(VariableStaticAction):
class PlaceObject(InteractWithGameObject):
class HarvestResource(InteractWithGameObject):
class Move(VariableStaticAction):




class DynamicAction(Action):
class LocateObject(DynamicAction):
class MoveToLocation(DynamicAction):
class ObtainResource(DynamicAction)
class AcquireResource(ObtainResource):
class AccessResource(ObtainResource):
class PickUpResource(DynamicAction):
class CompoundAction(DynamicAction):
