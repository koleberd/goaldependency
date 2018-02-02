from action import *
from gameState import *
from playerState import *
from playerStateTarget import *
from playerStateSolution import *
from actionTarget import *
from playerStateFactory import *
from actionFactory import *
from graphviz import *
from inventoryManager import *
import time
import random
from gameWorld2d import *
import gameController
import tensorflow as tf
import numpy as np
import dependencyTree as dtree
# action target selection functions
def selectCheapest(at_arr):
    cheapest = at_arr[0]
    for at in at_arr:
        if at.temp_cost_up < cheapest.temp_cost_up:
            cheapest = at
    return cheapest
def selectMostExpensive(at_arr):

    exp = at_arr[0]
    for at in at_arr:
        if at.temp_cost_up > exp.temp_cost_up:
            exp = at
    return exp
def selectRandom(at_arr):
    return at_arr[random.randint(0,len(at_arr)-1)]
def selectSequential(at_arr):
    min_id = at_arr[0]
    for at in at_arr:
        if id(min_id) > id(at):
            min_id = at
    #print(id(min_id))
    return min_id
def selectDeepest(at_arr):
    depth = [0 for x in range(0,len(at_arr))]
    for i in range(0,len(at_arr)):
        depth[i] = at_arr[i].getNodeDepth()
    sel = 0
    for i in range(0,len(at_arr)):
        if depth[sel] < depth[i]:
            sel = i
    return at_arr[sel]
def selectMostShallow(at_arr):
    sel = at_arr[0]
    for at in at_arr:
        if at.node_depth < sel.node_depth:
            sel = at
    return sel
def selectSmart(at_arr):
    #select a node with a highly reduced upward cost but a large downward cost
    return selectFirst(at_arr)
def selectUser(at_arr):
    return selectMostShallow(at_arr)
    for at in at_arr:
        if 'harvest' in at.act.name:
            return at
    for at in at_arr:
        if 'craft' in at.act.name and 'inv' not in at.act.name:
            return at
    for at in at_arr:
        if 'craft' in at.act.name:
            return at
    return at_arr[0]
    #select in order of priority
    #   harvest
    #   craft
    #   inv craft
    #   locate

def getWorldCosts(world,world_name):
    flnm = 'json/world_configs/'+world_name+'_costs.json'
    if not os.path.isfile(flnm):
        dists = world.getAverageDistances()
        with open(flnm,'w+') as fl:
            json.dump(dists,fl,indent=4)

    with open(flnm) as wldcf:
        return json.load(wldcf)

DISTANCE = 10
RAYS = 5
FOV_ANGLE = 120
KERNEL_RADIUS = 10
INPUT_DIM = (KERNEL_RADIUS*2+1)**2
OUTPUT_DIM = 6
bl_ind = {None:0,'wood':1,'stone':2,'crafting bench':3,'iron ore':4,'coal':5,'wall':6}

def weightVar(in_dim,out_dim):
    return tf.Variable(tf.truncated_normal([in_dim,out_dim], mean=.5, stddev=0.25))
def biasVar(in_dim):
    return tf.Variable(tf.zeros([in_dim]))
def deepnn(input_tensor):
    hidden1_dim = 256
    hidden2_dim = 256
    out_dim = OUTPUT_DIM
    dropout_rate = tf.placeholder(tf.float32)

    hidden1 = tf.matmul(input_tensor,weightVar(INPUT_DIM,hidden1_dim))
    hidden1 = tf.add(hidden1,biasVar(hidden1_dim))
    hidden1 = tf.nn.relu(hidden1)
    hidden2 = tf.matmul(hidden1,weightVar(hidden1_dim,hidden2_dim))
    hidden2 = tf.add(hidden2,biasVar(hidden2_dim))
    hidden2 = tf.nn.relu(hidden2)
    hidden2 = tf.nn.dropout(hidden2, dropout_rate)
    output_tensor = tf.matmul(hidden2,weightVar(hidden2_dim,out_dim))
    output_tensor = tf.add(output_tensor,biasVar(out_dim))
    output_tensor = tf.nn.relu(output_tensor)
    output_tensor = tf.div(
                        tf.subtract(
                            output_tensor,
                            tf.reduce_min(output_tensor)),
                        tf.subtract(
                            tf.reduce_max(output_tensor),
                            tf.reduce_min(output_tensor)))


    #input is n * (resource_type, distance) where n is number or rays
    #output is n * float with range [0:1]


    return output_tensor, dropout_rate

def preproc(stats,averages):
    ats = {}
    for i in range(0,len(stats)):
        el = stats[i]
        if el['at'] not in ats.keys():
            ats[el['at']] = {'type':el['type'],'inputs':[el['frame']]}
        else:
            ats[el['at']]['inputs'].append(el['frame'])






def run2d3d(config_name,select_method,select_name="",save_tree=False,save_path=False):
    full_start = time.time()

    with open(config_name) as jscf:
        config = json.load(jscf)
    world_2d = GameWorld2d( config['world_2d_location'],(config['spawn_pos'][0],config['spawn_pos'][1]))
    default_costs = getWorldCosts(world_2d,'rv_1')
    action_factory = ActionFactory(default_costs)
    level_index = dtree.decomposePS(PlayerState.parsePlayerStateJSON(config['target_ps']),config['simulation_name'],action_factory)
    inv_manager = InventoryManager()
    pm = PlayerMemory()
    gs = GameState(ps=None,pm=pm,fov=None,inv=inv_manager,world_2d=world_2d)

    root = level_index[0][0]
    scales = {} #action_factory.scaleCosts(gs.fov)
    root.calculateCostUp(scales)
    root.calculateCostSetDown([])
    root.calculateDepth(0)
    steps = [] #to keep track of the series of AT's as they're executed
    images = [] #used for gif rendering
    sim_output = []


    print('---- STARTING SIMUILATION  ----')
    print('selection method: ' + str(select_name))
    full_start2 = time.time()
    while(not root.isComplete()):
        #calculate cost scalars based on field of view
        #root.calculateCostUp(scales)
        #images.append(np.array(resize_no_blur(gs.world_2d.renderPath(gs.pm.metrics['path'][-10:]),2))) #uncomment if rendering a gif
        leaf_set = root.getLeafNodes()
        frame =world_2d.getKernel(KERNEL_RADIUS)
        selected_at = select_method(leaf_set) #level_index[0][0].select() #select at for execution
        if len(steps) == 0 or id(steps[-1]) != id(selected_at): #record selected AT
            steps.append(selected_at)
            #upwardPruneTree(level_index) #only need to prune if you're graphing the tree
            #downwardPruneTree(level_index)
            #graphTree(level_index,config['simulation_name'] + '_' + str(gs.world_step),selectedAT=selected_at)
        gs.pm.metrics['path'].append(gs.world_2d.pos)
        at_completed = selected_at.execute(gs)
        if at_completed: #execute AT
            root.calculateCostUp(scales)
        gs.world_step += 1
        sim_output.append({'frame':frame,'at':id(selected_at),'completed':at_completed,'type':selected_at.act.name})





    print(str(time.time()-full_start) + ' sec full run')
    print(str(time.time()-full_start2) + ' sec sim')
    print('steps taken: ' + str(len(gs.pm.metrics['path'])))
    '''
    print('rendering .gif')
    render_t = time.time()
    imageio.mimsave('simulation/' + select_name + '_animation.gif',images)
    print('rendered .gif in ' + str(time.time() - render_t) + ' sec')
    '''
    return sim_output

def main():
    learning_rate = 0
    training_rounds = 10
    input_tensor = tf.placeholder(tf.float32,shape=[None,INPUT_DIM],name='input_tensor')
    network_output,dropout_rate = deepnn(input_tensor)
    loss = tf.reduce_mean(tf.log(network_output))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss)

    average_costs = {}
    #run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectMostShallow(x),select_name='most shallow')
    #run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectUser(x),select_name='user')
    sim_out = run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectCheapest(x),select_name='cheapest')
    sim_out = preproc(sim_out,average_costs)
    #print(sim_out)
    '''
    with tf.Session() as sess:
        do = 'something'
    '''

#run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectSmart(x),select_name='smart')
main()

'''
bound to loop
run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectSequential(x),select_name='sequential')
run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectRandom(x),select_name='random')
run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectMostExpensive(x),select_name='most expensive')
run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectDeepest(x),select_name='deepest')
'''
