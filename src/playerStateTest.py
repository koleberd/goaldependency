from action import *
from gameState import *
from playerState import *
from playerStateTarget import *
from playerStateSolution import *
from actionTarget import *
from playerStateFactory import *
from actionFactory import *

def test():
    testHash()
    testPS()
    testPST()
    testActionFactory();
    print('------ tests concluded ------')

def testFulfill():
    do = 'nothing'

def testPST():
    tps = PlayerState(inventory={'wood':1,'stone':1})
    tpst = PlayerStateTarget(tps)
    #for ps in tpst.attributeList.keys():
    #    print(str(ps))

    tps2 = PlayerState(inventory={'wood':1})
    tps3 = PlayerState(inventory={'stone':1})

    assert(len(tpst.attributeList.keys()) == 2)
    assert(tps2 in tpst.attributeList.keys())
    assert(tps3 in tpst.attributeList.keys())

def testActionFactory():
    tps = PlayerState(inventory={'wood':1})
    fact = ActionFactory()
    acts = fact.getActions(tps)
    for act in acts:
        assert('wood' in act.ps_res.inventory.keys())



def testHash():

    tps = PlayerState(inventory={'wood':1})
    tps2 = PlayerState(inventory={'wood':1})
    tps3 = PlayerState(inventory={'wood':2})
    tps4 = PlayerState(inventory={'wood':1,'stone':1})
    tps5 = PlayerState(inventory={'wood':1},buffs={'speed':1})


    assert(hash(tps) == hash(tps2))
    assert(hash(tps) != hash(tps3))
    assert(hash(tps) != hash(tps4))
    assert(hash(tps) != hash(tps5))

def testPS():
    tps1 = PlayerState(inventory={'wood':1})
    tps2 = PlayerState(inventory={'wood':2})
    tps3 = PlayerState(inventory={'stone':1})
    tps4 = PlayerState(inventory={'wood':1,'stone':1})
    tps5 = PlayerState(inventory={'wood':3})


    assert(tps1 + tps2 == tps5)
    assert(tps1 + tps3 == tps4)

    assert(tps5 - tps2 == tps1)
    assert(tps5 - tps1 == tps2)
    assert(tps4 - tps1 == tps3)
    assert(tps4 - tps3 == tps1)

def testCyclicRequirement():
    tps1 = PlayerState(inventory={'wood':1})
    tps2 = PlayerState(inventory={'wood':1})
    tps4 = PlayerState(inventory={'wood axe':1})
    tps5 = PlayerState(inventory={'wood axe':10})
    tps6 = PlayerState(inventory={'wood plank':3})
    tps7 = PlayerState(inventory={'wood plank':4})
    tps8 = PlayerState(inventory={'wood':1})
    tps9 = PlayerState(inventory={'wood':1})

    tpst1 = PlayerStateTarget(tps1)                                       #wood
    tpss1 = PlayerStateSolution(tps2)                                     #wood via axe
    tat1 = ActionTarget(Action(tps4,tps2,1,'func1'))    #hit wood
    tpst1.addSolution(tps2,tpss1)
    tpss1.addParent(tpst1)
    tpss1.addchild(tat1)
    tat1.addParent(tpss1)

    tpst2 = PlayerStateTarget(tps4)
    tpss2 = PlayerStateSolution(tsp5)
    tat2 = ActionTarget(Action(tps8,tps5,1,'func2'))
    tpst2.addParent(tat1)
    tpst2.addSolution(tps5,tpss2)
    tpss2.addParent(tpst2)
    tpss2.addChild(tat2)
    tat2.addParent(tpss2)

    tpst3 = PlayerStateTarget(tps8)
    tpss2 = PlayerStateSolution(tps9)
    tat3 = ActionTarget(Action(PlayerState(),tps9,1,'func3'))
    tpst3.addParent(tat2)
    tpst3.addSolution(tps9,tpss3)
    tpss3.addParent(tpst3)
    tpss3.addChild(tat3)
    tat3.addParent(tpss3)

    assert(tat3.isCyclicRequirement(tps9))
    assert(tat2.isCyclicRequirement(tps9))












def testSomething():
    do = 'nothing'
