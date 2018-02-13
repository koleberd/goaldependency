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
bl_ind = {None:0,'wood':1,'stone':2,'crafting bench':3,'iron ore':4,'coal':5,'furnace':6,'wall':7} #for encoding data for input into nn
action_set = ['locateObject:wood','locateObject:stone','locateObject:crafting bench','locateObject:iron ore','locateObject:coal'] #for decoding data from output of nn

#--- NETWORK PARAMETERS ---
KERNEL_RADIUS = 10
INPUT_DIM = (KERNEL_RADIUS*2+1)**2
hidden1_dim = 256
hidden2_dim = 256

#--- TRAINING PARAMETERS ---
FRAME_SKIP_THRESHOLD = .45 #pick frame if random number [0,1] under threshold
FRAME_DENSITY_THRESHOLD = .09 #pick frame if frame density over threshold (frame density is ratio of non-empty units to all units in input kernel)
MOVING_AVERAGE_SIZE = 10 #moving average is compared to current run to determine if current run should be positively or negatively reinforced
OUTPUT_LOWER_BIAS = .25 #lowest possible output weight (max is always 1)
DROPOUT_TRAINING = 1.0 #chance a node will be kept during dropout during training
TRAINGING_ROUNDS = 200 #number of rounds to train
LEARNING_RATE = 0.01
BATCH_CUTOFF = 2000
VALIDATION_ROUNDS = 20
CROSS_BENCHMARK_SAMPLES = 20

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

def getWorldDensities():
    simulation_config_name = sim_config_dir + sim_name + '.json'
    with open(simulation_config_name) as simjs:
        simcf = json.load(simjs)
    world = GameWorld2d(simcf['world_2d_location'],spawn_random=False)

    return world.getDensities()

# action target selection functions
def selectMostShallow(at_arr,frame,prev,prev_time):
    sel = at_arr[0]
    for at in at_arr:
        if at.node_depth < sel.node_depth:
            sel = at
    return sel
def selectCheapest(at_arr,frame,prev,prev_time):
    cheapest = at_arr[0]
    for at in at_arr:
        if at.temp_cost_up < cheapest.temp_cost_up:
            cheapest = at
    return cheapest
def selectUser(at_arr,frame,prev,prev_time):
    #select in order of priority
    #   harvest
    #   craft
    #   inv craft
    #   locate
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
def selectCheapestDNN(at_arr,frame,prev,prev_time,nn_out):
    scaled = [float(at.temp_cost_up) for at in at_arr]

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

    selected = sorted(mins_set,key=lambda s: s.node_depth,reverse=True)[0] #pick the deepest node

    if prev != None and selected.act.name == prev.act.name: #action identical to previous selected, execute previous
        print(prev)
        print(selected)
        selected = prev
    elif prev != None and selected.act.name != prev.act.name: #introduce switch cost
        prev_cost_remaining = prev.temp_cost_up - prev_time
        prev_cost_multi = 1
        if prev.act.name in action_set:
            w_ind = action_set.index(prev.act.name)
            if not math.isnan(weights[w_ind]):
                prev_cost_multi = weights[w_ind]
        cost_switch = mincst + (prev_time*prev_cost_multi)
        cost_remaining = prev_cost_remaining * prev_cost_multi
        if cost_switch > cost_remaining:
            return prev

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
                if random.random() < FRAME_SKIP_THRESHOLD and frame_density > FRAME_DENSITY_THRESHOLD:
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

def run2d3d(config_name,select_method,select_name='',save_tree=False,save_path=False,random_spawn=False,spawn_pos=(0,0)):
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

    world_2d = GameWorld2d( config['world_2d_location'],spawn_random=random_spawn,spawn_pos=spawn_pos)# (config['spawn_pos'][0],config['spawn_pos'][1]))
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
    steps_this_act = 0
    while(not root.isComplete()):
        #calculate cost scalars based on field of view
        #root.calculateCostUp(scales)
        #images.append(np.array(resize_no_blur(gs.world_2d.renderPath(gs.pm.metrics['path'][-10:]),2))) #uncomment if rendering a gif
        leaf_set = root.getLeafNodes()
        frame = world_2d.getKernel(KERNEL_RADIUS)
        selected_at = select_method(leaf_set,frame,prev_at,steps_this_act) #level_index[0][0].select() #select at for execution
        if len(steps) == 0 or id(steps[-1]) != id(selected_at): #record selected AT
            steps_this_act = 0
            steps.append(selected_at)
            #print(selected_at.act.name+ '-' + str(id(selected_at)))
            #dtree.upwardPruneTree(level_index) #only need to prune if you're graphing the tree - BUGGED
            #dtree.downwardPruneTree(level_index)
            #dtree.graphTree(level_index,config['simulation_name'] + '_' + str(gs.world_step),selectedAT=selected_at)
        else:
            steps_this_act += 1

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

def train():
    simulation_config_name = sim_config_dir + sim_name + '.json'

    #set up NN
    input_tensor = tf.placeholder(tf.float32,shape=[None,INPUT_DIM],name='input_tensor')
    label_tensor = tf.placeholder(tf.float32, [None, len(action_set)],name='label_tensor')
    output_tensor,dropout_rate = deepnn(input_tensor)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_tensor,logits=output_tensor))#tf.reduce_mean(tf.log(outout_tensor))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_step = optimizer.minimize(loss)

    #load world config
    with open(simulation_config_name) as simjs:
        simcf = json.load(simjs)

     #load world costs
    temp_world = GameWorld2d(simcf['world_2d_location'],spawn_random=False)
    average_costs = getWorldCosts(temp_world,simcf['simulation_name'])
    validation_pos = [temp_world.randomizePos() for x in range(VALIDATION_ROUNDS)]

    #stats = []
    moving_averages = [[average_costs[x] for x in action_set]]
    with tf.Session() as sess:
        training_performance = []
        total_time = time.time()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        tf.add_to_collection('n_input',input_tensor)
        tf.add_to_collection('n_output',output_tensor)
        tf.add_to_collection('n_dropout',dropout_rate)

        #--- INITIAL BENCHMARK ---
        init_avg = 0
        init_valid_rounds = 0
        init_time_sum = 0
        for step in range(VALIDATION_ROUNDS):

            sim_out,sim_len,sim_time = run2d3d(simulation_config_name,select_method = lambda x,frame,prev,prev_time: selectCheapestDNN(x,frame,prev,prev_time,output_tensor),select_name='cheapest',random_spawn=False,spawn_pos=validation_pos[step])
            print('Benchmarking... round ' + str(step+1) + '/' + str(VALIDATION_ROUNDS) + ' - ' + str(sim_len))
            init_time_sum += sim_time
            if sim_len != -1:
                init_avg += sim_len
                init_valid_rounds += 1

        init_avg = float(init_avg)/float(init_valid_rounds)
        print('initial average performance: ' + str(init_avg) + ' over ' + str(init_valid_rounds) + ' rounds')
        est_train_time = ((float(init_time_sum)/float(VALIDATION_ROUNDS)) * float(TRAINGING_ROUNDS)) / 60.0
        print(str(init_time_sum).split('.')[0] + 's to execute initial test.\n Expect ' + str(est_train_time)[:7] + 'min to train, plus another ' + str(float(init_time_sum)/60.0)[:7] + 'min to validate')

        #--- TRAINING ---
        for step in range(0,TRAINGING_ROUNDS):
            sim_out,sim_len,sim_time = run2d3d(simulation_config_name,select_method = lambda x,frame,prev,prev_time: selectCheapestDNN(x,frame,prev,prev_time,output_tensor),select_name='cheapest',random_spawn=RAND_SPAWN)

            if sim_len == -1:
                print('Bad batch')
                continue


            #calculate moving average to compare current performance against
            current_averages = [0 for x in range(len(action_set))]
            for x in moving_averages:
                for i in range(len(action_set)):
                    current_averages[i] += x[i]
            current_averages = [float(x)/float(len(moving_averages)) for x in current_averages]
            batch, sim_averages = generateBatch(sim_out,current_averages)
            moving_averages = moving_averages[1:]
            moving_averages.append(sim_averages)


            #stats.append({'samples':len(batch[0]),'world cycles':sim_len,'batch num':step,'sim time':sim_time})
            #print(('training on batch ' + str(step) + ' with ' + str(len(batch[0])) + ' samples (sim world cycles: ' + str(sim_len) + ') ' + str(batch[1][0])).encode('utf-8').decode('ascii'))

            current_loss = loss.eval(feed_dict={input_tensor: batch[0], label_tensor: batch[1], dropout_rate: 1.0})
            print('training... round ' + str(step+1) + '/' + str(TRAINGING_ROUNDS) + ' - performance: ' + str(sim_len) + ' - samples: ' + str(len(batch[0])) + ' - loss: ' + str(current_loss))
            training_performance.append({'performance':sim_len,'loss':float(current_loss)})
            train_step.run(feed_dict={input_tensor: batch[0], label_tensor: batch[1], dropout_rate: DROPOUT_TRAINING})

        #--- FINAL BENCHMARK ---
        end_avg = 0
        end_valid_rounds = 0
        end_time_sum = 0
        for step in range(VALIDATION_ROUNDS):

            sim_out,sim_len,sim_time = run2d3d(simulation_config_name,select_method = lambda x,frame,prev,prev_time: selectCheapestDNN(x,frame,prev,prev_time,output_tensor),select_name='cheapest',random_spawn=False,spawn_pos=validation_pos[step])
            print('Validating... round ' + str(step+1) + '/' + str(VALIDATION_ROUNDS) + ' - ' + str(sim_len))
            end_time_sum += sim_time
            if sim_len != -1:
                end_avg += sim_len
                end_valid_rounds += 1

        #--- OUTPUT STATS ---
        total_time = time.time() - total_time
        end_avg = float(end_avg)/float(end_valid_rounds)
        improvement = end_avg/init_avg
        print('final average performance: ' + str(end_avg) + ' over ' + str(end_valid_rounds) + ' rounds')
        print('total time elapsed: ' + str(total_time) + ' s')
        print('performance increase of ' + str(improvement))
        data = {'stats':training_performance,
                'init benchmark':init_avg,
                'final benchmark':end_avg,
                'benchmark size':VALIDATION_ROUNDS,
                'training size':TRAINGING_ROUNDS,
                'world configuration':simulation_config_name,
                'dropout':DROPOUT_TRAINING,
                'frame skip threshold':FRAME_SKIP_THRESHOLD,
                'frame density threshold':FRAME_DENSITY_THRESHOLD,
                'hidden 1 dim': hidden1_dim,
                'hidden 2 dim': hidden2_dim,
                'performance improvement':improvement,
                'random spawn during training':str(RAND_SPAWN)
                }
        with open(sim_stat_dir + sim_name + '-' + str(time.time())[5:10] + '.json','w+') as output_json:
            json.dump(data,output_json,indent=4,sort_keys=True)



        #--- SAVE MODEL IF EFFECTIVE ---
        if improvement < 1.0:
            m_name = sim_name + '_' + str(improvement)
            print('Saving model "' + m_name + '"')
            saver.save(sess,'trainedModels/' + m_name)
            m_name = 'trainedModels/' + m_name
            benchmarkAgainstAlternates(m_name)
        return improvement

def benchmarkAgainstAlternates(model_name):
    simulation_config_name = sim_config_dir + sim_name + '.json'

    #load world config
    with open(simulation_config_name) as simjs:
        simcf = json.load(simjs)

     #load world costs
    temp_world = GameWorld2d(simcf['world_2d_location'],spawn_random=False)
    average_costs = getWorldCosts(temp_world,simcf['simulation_name'])
    sample_locs = [temp_world.randomizePos() for x in range(CROSS_BENCHMARK_SAMPLES)]


    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_name + '.meta')
        saver.restore(sess,model_name)
        n_input = tf.get_collection('n_input')[0]
        n_output = tf.get_collection('n_output')[0]
        n_dropout = tf.get_collection('n_dropout')[0]
        selection_set = {
            'cheapestDNN':lambda x,frame,prev,prev_time: selectCheapestDNN(x,frame,prev,prev_time,n_output),
            'cheapest':lambda x,frame,prev,prev_time: selectCheapest(x,frame,prev,prev_time),
            'shallow':lambda x,frame,prev,prev_time: selectMostShallow(x,frame,prev,prev_time),
            'user':lambda x,frame,prev,prev_time: selectUser(x,frame,prev,prev_time)
        }
        for s_name in selection_set:
            avg = 0
            count = 0
            for loc in sample_locs:

                sim_out,sim_len,sim_time = run2d3d(simulation_config_name,select_method = selection_set[s_name],select_name=s_name,random_spawn=False,spawn_pos=loc)
                print('running ' + str(count+1) + '/' + str(len(sample_locs)) + ' for ' + s_name + ' - ' + str(sim_len))
                if sim_len != -1:
                    avg += sim_len
                    count += 1
            avg = float(avg)/float(count)
            print(s_name + ': ' + str(avg))


#run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectMostShallow(x),select_name='most shallow')
#run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x: selectUser(x),select_name='user')
#sim = run2d3d('json/simulation_configs/rv_1.json',select_method = lambda x,f: selectCheapest(x,f),select_name='cheapest')
#print(sim[1])
#train()
#benchmarkAgainstAlternates('trainedModels/rv_2_0.5392337205024439')
#getWorldDensities()
benchmarkAgainstAlternates('trainedModels/rv_3_0.7041499330655957')
