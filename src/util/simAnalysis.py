import json
import os

sim_dir = 'json/simulation_stats'
files = os.listdir(sim_dir)

files = [sim_dir + '/' + f for f in files]
for f in files:
    with open(f) as sim_js:
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

    with open(f,'w+') as sim_js:
        json.dump(sim_stats,sim_js,indent=4,sort_keys=True)
