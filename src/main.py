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
import sys
import math




#gets average distance for every object from every open position for a world, and puts it in a json file
def getWorldCosts(world,world_name):
    flnm = 'json/world_configs/'+world_name+'_costs.json'
    if not os.path.isfile(flnm):
        dists = world.getAverageDistances()
        dists2 = {}
        for dk in dists:
            name = 'locateObject:' + dk
            #name = dk.split(' ')
            #name = 'locate' + ''.join([x[:1].capitalize() + x[1:] for x in name])
            dists2[name] = dists[dk]

        with open(flnm,'w+') as fl:
            json.dump(dists2,fl,indent=4)

    with open(flnm) as wldcf:
        return json.load(wldcf)


DISTANCE = 10
RAYS = 5
FOV_ANGLE = 120
KERNEL_RADIUS = 10
INPUT_DIM = (KERNEL_RADIUS*2+1)**2
OUTPUT_DIM = 6
FRAME_SKIP_THRESHOLD = .35

bl_ind = {None:0,'wood':1,'stone':2,'crafting bench':3,'iron ore':4,'coal':5,'wall':6}
action_set = ['locateObject:wood','locateObject:stone','locateObject:crafting bench','locateObject:iron ore','locateObject:coal']

# action target selection functions
'''
def selectMostExpensive(at_arr,frame):

    exp = at_arr[0]
    for at in at_arr:
        if at.temp_cost_up > exp.temp_cost_up:
            exp = at
    return exp
def selectRandom(at_arr,frame):
    return at_arr[random.randint(0,len(at_arr)-1)]
def selectSequential(at_arr,frame):
    min_id = at_arr[0]
    for at in at_arr:
        if id(min_id) > id(at):
            min_id = at
    #print(id(min_id))
    return min_id
def selectDeepest(at_arr,frame):
    depth = [0 for x in range(0,len(at_arr))]
    for i in range(0,len(at_arr)):
        depth[i] = at_arr[i].getNodeDepth()
    sel = 0
    for i in range(0,len(at_arr)):
        if depth[sel] < depth[i]:
            sel = i
    return at_arr[sel]
'''
def selectMostShallow(at_arr,frame,prev):
    sel = at_arr[0]
    for at in at_arr:
        if at.node_depth < sel.node_depth:
            sel = at
    return sel
def selectCheapest(at_arr,frame,prev):
    cheapest = at_arr[0]
    for at in at_arr:
        if at.temp_cost_up < cheapest.temp_cost_up:
            cheapest = at
    return cheapest
#select in order of priority
#   harvest
#   craft
#   inv craft
#   locate
def selectUser(at_arr,frame,prev):
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

def selectCheapestDNN(at_arr,frame,prev,nn_in,nn_out):
    scaled = [float(at.temp_cost_up) for at in at_arr]

    #nn_in = frame
    weights = nn_out.eval(feed_dict={'input_tensor:0':[frame],'dropout_rate:0':1.0})[0]
    #print(weights)
    for i in range(0,len(at_arr)):
        a_n = at_arr[i].act.name
        if a_n in action_set:
            w_ind = action_set.index(a_n)
            if not math.isnan(weights[w_ind]):
                scaled[i] *= weights[w_ind]

    #cheapest_ind = scaled.index(min(scaled))
    #selected = at_arr[cheapest_ind]

    mincst = min(scaled)
    mins_set = []
    for i in range(0,len(scaled)):
        if scaled[i] == mincst:
            mins_set.append(at_arr[i])

    selected = sorted(mins_set,key=lambda s: s.node_depth,reverse=True)[0]

    if prev != None and selected.act.name == prev.act.name:
        print(prev)
        print(selected)
        selected = prev

    return selected

def weightVar(in_dim,out_dim):
    return tf.Variable(tf.truncated_normal([in_dim,out_dim], mean=0.5, stddev=0.25))
def biasVar(in_dim):
    return tf.Variable(tf.zeros([in_dim]))
def deepnn(input_tensor):
    hidden1_dim = 256
    hidden2_dim = 256
    out_dim = len(action_set)
    dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

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
    #normalize to [0:1]
    output_tensor = tf.div(
                        tf.subtract(
                            output_tensor,
                            tf.reduce_min(output_tensor)),
                        tf.subtract(
                            tf.reduce_max(output_tensor),
                            tf.reduce_min(output_tensor)))

    #normalize to [.25:1]
    final_bias = [.25 for x in range(0,out_dim)]
    output_tensor = tf.scalar_mul(.75, tf.add(output_tensor,final_bias))


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

    for at_k in ats:
        at = ats[at_k]

        at['output'] = [0.0 if at['type'] == action_set[x] else 1.0 for x in range(0,len(action_set))]
        if at['type'] in averages.keys() and len(at['inputs']) > averages[at['type']]:
            at['output'] = [max(1.0,x) for x in at['output']]
    return ats
def constructBatch(stats):
    inputs = []
    logits = []
    for at_k in stats:
        at = stats[at_k]
        for ip in at['inputs']:
            if random.random() < FRAME_SKIP_THRESHOLD:
                inputs.append(ip)
                logits.append(at['output'])
    return (inputs,logits)

def run2d3d(config_name,select_method,select_name="",save_tree=False,save_path=False):
    full_start = time.time()
    PRINT = False
    with open(config_name) as jscf:
        config = json.load(jscf)
    world_2d = GameWorld2d( config['world_2d_location'],spawn_random=True)# (config['spawn_pos'][0],config['spawn_pos'][1]))
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

    print('---- STARTING SIMUILATION  ----') if PRINT else None
    print('selection method: ' + str(select_name)) if PRINT else None
    full_start2 = time.time()
    prev_at = None
    while(not root.isComplete()):
        #calculate cost scalars based on field of view
        #root.calculateCostUp(scales)
        #images.append(np.array(resize_no_blur(gs.world_2d.renderPath(gs.pm.metrics['path'][-10:]),2))) #uncomment if rendering a gif
        leaf_set = root.getLeafNodes()
        frame = world_2d.getKernel(KERNEL_RADIUS)
        selected_at = select_method(leaf_set,frame,prev_at) #level_index[0][0].select() #select at for execution
        if len(steps) == 0 or id(steps[-1]) != id(selected_at): #record selected AT
            steps.append(selected_at)
            #print(selected_at.act.name+ '-' + str(id(selected_at)))


            #dtree.upwardPruneTree(level_index) #only need to prune if you're graphing the tree - BUGGED
            #dtree.downwardPruneTree(level_index)
            #dtree.graphTree(level_index,config['simulation_name'] + '_' + str(gs.world_step),selectedAT=selected_at)
        gs.pm.metrics['path'].append(gs.world_2d.pos)
        at_completed = selected_at.execute(gs)
        if at_completed: #execute AT
            prev_at = None
            root.calculateCostUp(scales)
        gs.world_step += 1
        sim_output.append({'frame':frame,'at':id(selected_at),'completed':at_completed,'type':selected_at.act.name})
        if gs.world_step > 1500:
            gs.world_step = -1
            break
        #print('.',end='')
        #sys.stdout.flush()

    print(str(time.time()-full_start) + ' sec full run') if PRINT else None
    print(str(time.time()-full_start2) + ' sec sim') if PRINT else None
    print('steps taken: ' + str(len(gs.pm.metrics['path']))) if PRINT else None
    '''
    print('rendering .gif')
    render_t = time.time()
    imageio.mimsave('simulation/' + select_name + '_animation.gif',images)
    print('rendered .gif in ' + str(time.time() - render_t) + ' sec')
    '''
    sim_len = gs.world_step#len(gs.pm.metrics['path'])
    return sim_output,sim_len

def main():
    learning_rate = 0
    training_rounds = 500

    input_tensor = tf.placeholder(tf.float32,shape=[None,INPUT_DIM],name='input_tensor')
    label_tensor = tf.placeholder(tf.float32, [None, len(action_set)])
    output_tensor,dropout_rate = deepnn(input_tensor)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_tensor,logits=output_tensor))#tf.reduce_mean(tf.log(outout_tensor))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss)

    with open('json/world_configs/rv_1_costs.json') as cstjs:
        average_costs = json.load(cstjs)

    '''
    for x in sim_out:
        at = sim_out[x]
        if 0 in at['output']:
            print(at['type'] + '\t|\t' + str(at['output']))
    '''
    stats = []
    total_samples = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_time = time.time()
        for step in range(0,training_rounds):
            sim_time = time.time()
            sim_out,sim_len = run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x,frame,prev: selectCheapestDNN(x,frame,prev,input_tensor,output_tensor),select_name='cheapest')
            sim_time = time.time() - sim_time
            if sim_len == -1:
                print('Bad batch')
                return False
            sim_out = preproc(sim_out,average_costs)
            '''
            print('.',end='')
            sys.stdout.flush()
            '''
            batch = constructBatch(sim_out)
            total_samples += len(batch[0])
            stats.append({'samples':len(batch[0]),'world cycles':sim_len,'batch num':step,'sim time':sim_time})
            print(('training on batch ' + str(step) + ' with ' + str(len(batch[0])) + ' samples (sim world cycles: ' + str(sim_len) + ')').encode("utf-8").decode("ascii"))
            sys.stdout.flush()
            '''
            BATCH_SIZE = len(batch_set[0])
            trainingSet = tf.contrib.data.Dataset.from_tensor_slices(batch_set)
            next_element_training = trainingSet.make_one_shot_iterator().get_next()
            '''
            train_step.run(feed_dict={input_tensor: batch[0], label_tensor: batch[1], dropout_rate: 0.5})
    print('total samples trained: ' + str(total_samples))
    training_time = time.time() - training_time
    analysis = {}
    analysis['training time'] = training_time
    analysis['time per simulation'] = training_time / float(len(stats))
    with open('json/simulation_stats/rv_1_' + str(time.time()) + 'stats.json','w+') as ojs:
        json.dump({'stats':stats,'analysis':analysis},ojs,indent=4,sort_keys=True)


#run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectMostShallow(x),select_name='most shallow')
#run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectUser(x),select_name='user')
#sim = run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x,f: selectCheapest(x,f),select_name='cheapest')
#print(sim[1])
main()

'''
bound to loop
run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectSequential(x),select_name='sequential')
run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectRandom(x),select_name='random')
run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectMostExpensive(x),select_name='most expensive')
run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectDeepest(x),select_name='deepest')
'''
