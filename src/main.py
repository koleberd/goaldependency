from action import *
from gameState import *
from playerState import *
from playerStateTarget import *
from playerStateSolution import *
from actionTarget import *
from playerStateFactory import *
from actionFactory import *


def decomposePS(ps, parentPSS, parentPST, parentAT):
    pst = PlayerStateTarget(ps)
    pst.addParent(parentAT)
    for attr in pst.attributeList:
        for act in ActionFactory().getActions(attr):
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

def prunePST(pst):
    do = 'nothing'


toplvlps = PlayerState()
#add things to the top level ps
toplvlps.inventory['cobblestone'] = 200
toplvlpst = decomposePS(toplvlps,None,None,None)
prunePST(toplvlpst)
