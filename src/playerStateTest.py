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
    testPST()
    testActionFactory();


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
