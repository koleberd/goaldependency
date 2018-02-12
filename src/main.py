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





#--- SIMULATION PARAMETERS ---
RAND_SPAWN = True
sim_name = 'rv_2'
bl_ind = {None:0,'wood':1,'stone':2,'crafting bench':3,'iron ore':4,'coal':5,'wall':6}
action_set = ['locateObject:wood','locateObject:stone','locateObject:crafting bench','locateObject:iron ore','locateObject:coal']

#--- NETWORK PARAMETERS ---
KERNEL_RADIUS = 10
INPUT_DIM = (KERNEL_RADIUS*2+1)**2
hidden1_dim = 256
hidden2_dim = 256

#--- TRAINING PARAMETERS ---
FRAME_SKIP_THRESHOLD = .25 #pick frame if random number [0,1] under threshold
FRAME_DENSITY_THRESHOLD = .1 #pick frame if frame density over threshold (frame density is ratio of non-empty units to all units in input kernel)
MOVING_AVERAGE_SIZE = 10 #moving average is compared to current run to determine if current run should be positively or negatively reinforced
OUTPUT_LOWER_BIAS = .15 #lowest possible output weight (max is always 1)
DROPOUT_TRAINING = .5 #chance a node will be kept during dropout during training
TRAINGING_ROUNDS = 1000 #number of rounds to train
LEARNING_RATE = 0.1
BATCH_CUTOFF = 2000
VALIDATION_ROUNDS = 100

#--- CONFIGURATION CONSTANTS ---
sim_stat_dir = 'json/simulation_stats/'
sim_config_dir = 'json/simulation_configs/'


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

# action target selection functions
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
def selectUser(at_arr,frame,prev):
    #select in order of priority
    #   harvest
    #   craft
    #   inv craft
    #   locate
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
    final_bias = [OUTPUT_LOWER_BIAS for x in range(0,out_dim)]
    output_tensor = tf.scalar_mul((1.0 - OUTPUT_LOWER_BIAS), tf.add(output_tensor,final_bias))



    return output_tensor, dropout_rate

def frameDensity(frame):
    total = 0
    for i in frame:
        if i != 0:
            total += 1
    return float(total)/float(len(frame))
def generateBatch(stats,averages):
    #group identical ats
    OUTPUT_CLASSES = len(action_set)
    ats = {}
    for i in range(0,len(stats)):
        el = stats[i]
        if el['at'] not in ats.keys():
            ats[el['at']] = {'type':el['type'],'inputs':[el['frame']]}
        else:
            ats[el['at']]['inputs'].append(el['frame'])

    sim_sums = [x for x in range(OUTPUT_CLASSES)]
    sim_counts = [x for x in range(OUTPUT_CLASSES)]

    inputs = []
    for atk in ats.keys():
        at = ats[atk]
        if at['type'] in action_set:
            ind = action_set.index(at['type'])
            sim_sums[ind] += len(at['inputs'])
            sim_counts[ind] += 1
            for ip in at['inputs']:
                frame_density = frameDensity(ip)
                #print(frame_density)
                if random.random() < FRAME_SKIP_THRESHOLD or frame_density > FRAME_DENSITY_THRESHOLD:
                    inputs.append(ip)

    sim_averages = [float(sim_sums[x]) / float(sim_counts[x]) for x in range(OUTPUT_CLASSES)]
    #print(sim_averages)
    #print(averages)
    sim_label = [0 if sim_averages[x] < averages[x] else 1 for x in range(OUTPUT_CLASSES)]



    labels = [sim_label for x in range(0,len(inputs))]
    return (inputs,labels),sim_averages

def generateAnalysis(in_flnm,out_flnm):
    with open(in_flnm) as sim_js:
        sim_stats = json.load(sim_js)
    analysis = {}
    seg_size = int(len(sim_stats['stats'])/10)
    segs = [0 for x in range(0,10)]
    for x in range(0,10):
        seg_sum = 0
        for s in range(0,seg_size):
            seg_sum += sim_stats['stats'][x*seg_size+s]['world cycles']
        segs[x] = seg_sum/seg_size
    sim_stats['analysis']['moving average'] = segs
    sim_stats['analysis']['total samples'] = 0
    sim_stats['analysis']['total simulations'] = len(sim_stats['stats'])
    for s in sim_stats['stats']:
        sim_stats['analysis']['total samples'] += s['samples']

    with open(out_flnm,'w+') as sim_js:
        json.dump(sim_stats,sim_js,indent=4,sort_keys=True)



def run2d3d(config_name,select_method,select_name='',save_tree=False,save_path=False,random_spawn=False):
    '''
    Returns arr,int,float
        where arr is in form of [{'frame':[],'at':id,'completed':bool,'type':string},
                                ...]
        and int is the number of world cycles
        and float is the millis to execute simulation

    '''
    full_start = time.time()
    PRINT = False
    with open(config_name) as jscf:
        config = json.load(jscf)
    world_2d = GameWorld2d( config['world_2d_location'],spawn_random=random_spawn)# (config['spawn_pos'][0],config['spawn_pos'][1]))
    default_costs = getWorldCosts(world_2d,config['simulation_name'])
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
        if gs.world_step > BATCH_CUTOFF:
            gs.world_step = -1
            break
        #print('.',end='')
        #sys.stdout.flush()
    full_start = time.time() - full_start
    print(str(full_start) + ' sec full run') if PRINT else None
    print(str(time.time()-full_start2) + ' sec sim') if PRINT else None
    print('steps taken: ' + str(len(gs.pm.metrics['path']))) if PRINT else None
    '''
    print('rendering .gif')
    render_t = time.time()
    imageio.mimsave('simulation/' + select_name + '_animation.gif',images)
    print('rendered .gif in ' + str(time.time() - render_t) + ' sec')
    '''

    sim_len = gs.world_step#len(gs.pm.metrics['path'])
    return sim_output,sim_len,full_start

def main():


    #set up NN
    input_tensor = tf.placeholder(tf.float32,shape=[None,INPUT_DIM],name='input_tensor')
    label_tensor = tf.placeholder(tf.float32, [None, len(action_set)])
    output_tensor,dropout_rate = deepnn(input_tensor)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_tensor,logits=output_tensor))#tf.reduce_mean(tf.log(outout_tensor))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_step = optimizer.minimize(loss)

    simulation_config_name = sim_config_dir + sim_name + '.json'


    #load world config
    with open(simulation_config_name) as simjs:
        simcf = json.load(simjs)

     #load world costs
    temp_world = GameWorld2d(simcf['world_2d_location'],spawn_random=False)
    average_costs = getWorldCosts(temp_world,simcf['simulation_name'])

    #run benchmark set for validation




    stats = []
    moving_averages = [[average_costs[x] for x in action_set]]
    with tf.Session() as sess:
        training_performance = []
        total_time = time.time()
        sess.run(tf.global_variables_initializer())

        #--- INITIAL BENCHMARK ---
        init_avg = 0
        init_valid_rounds = 0
        init_time_sum = 0
        for step in range(VALIDATION_ROUNDS):
            print('Benchmarking... round ' + str(step+1) + '/' + str(VALIDATION_ROUNDS))
            sim_out,sim_len,sim_time = run2d3d(simulation_config_name,select_method = lambda x,frame,prev: selectCheapestDNN(x,frame,prev,input_tensor,output_tensor),select_name='cheapest',random_spawn=RAND_SPAWN)
            init_time_sum += sim_time
            if sim_len != -1:
                init_avg += sim_len
                init_valid_rounds += 1

        init_avg = float(init_avg)/float(init_valid_rounds)
        print('initial average performance: ' + str(init_avg) + ' over ' + str(init_valid_rounds) + ' rounds')
        est_train_time = ((float(init_time_sum)/float(VALIDATION_ROUNDS)) * float(TRAINGING_ROUNDS)) / 60.0
        print(str(init_time_sum).split('.')[0] + 's to execute initial test.\n Expect ' + str(est_train_time)[:7] + 'min to train, plus another ' + str(float(init_time_sum)/60.0)[:7] + 'min to validate')




        training_time = time.time()
        for step in range(0,TRAINGING_ROUNDS):

            sim_out,sim_len,sim_time = run2d3d(simulation_config_name,select_method = lambda x,frame,prev: selectCheapestDNN(x,frame,prev,input_tensor,output_tensor),select_name='cheapest',random_spawn=RAND_SPAWN)
            training_performance.append(sim_len)
            if sim_len == -1:
                print('Bad batch')
                continue
            current_averages = [0 for x in range(len(action_set))]



            for x in moving_averages:
                for i in range(len(action_set)):
                    current_averages[i] += x[i]
            current_averages = [float(x)/float(len(moving_averages)) for x in current_averages]

            batch, sim_averages = generateBatch(sim_out,current_averages)
            moving_averages = moving_averages[1:]
            moving_averages.append(sim_averages)

            stats.append({'samples':len(batch[0]),'world cycles':sim_len,'batch num':step,'sim time':sim_time})
            #print(('training on batch ' + str(step) + ' with ' + str(len(batch[0])) + ' samples (sim world cycles: ' + str(sim_len) + ') ' + str(batch[1][0])).encode('utf-8').decode('ascii'))
            print('training... round ' + str(step+1) + '/' + str(TRAINGING_ROUNDS))

            train_step.run(feed_dict={input_tensor: batch[0], label_tensor: batch[1], dropout_rate: DROPOUT_TRAINING})

        #--- FINAL BENCHMARK ---
        end_avg = 0
        end_valid_rounds = 0
        end_time_sum = 0
        for step in range(VALIDATION_ROUNDS):
            print('Validating... round ' + str(step+1) + '/' + str(VALIDATION_ROUNDS))
            sim_out,sim_len,sim_time = run2d3d(simulation_config_name,select_method = lambda x,frame,prev: selectCheapestDNN(x,frame,prev,input_tensor,output_tensor),select_name='cheapest',random_spawn=RAND_SPAWN)
            end_time_sum += sim_time
            if sim_len != -1:
                end_avg += sim_len
                end_valid_rounds += 1
        total_time = time.time() - total_time
        end_avg = float(end_avg)/float(end_valid_rounds)
        print('final average performance: ' + str(end_avg) + ' over ' + str(end_valid_rounds) + ' rounds')
        print('total time elapsed: ' + str(total_time) + ' s')

        with open(sim_stat_dir + sim_name + '-' + str(time.time())[5:11] + '.json','w+') as output_json:
            json.dump({'stats':training_performance,'init benchmark':init_avg,'final benchmark':end_avg,'benchmark size':VALIDATION_ROUNDS,'training size':TRAINGING_ROUNDS,'world configuration':simulation_config_name},output_json,indent=4,sort_keys=True)


    '''
    training_time = time.time() - training_time
    analysis = {}
    analysis['training time'] = training_time
    analysis['time per simulation'] = training_time / float(len(stats))
    sim_stat_dir = 'json/simulation_stats/'


    with open('json/simulation_stats/' + sim_name + '_' + str(time.time()) + 'stats.json','w+') as ojs:
        json.dump({'stats':stats,'analysis':analysis},ojs,indent=4,sort_keys=True)
    generateAnalysis('json/simulation_stats'/)
    '''


#run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectMostShallow(x),select_name='most shallow')
#run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectUser(x),select_name='user')
#sim = run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x,f: selectCheapest(x,f),select_name='cheapest')
#print(sim[1])
main()
