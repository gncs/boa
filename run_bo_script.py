import numpy as np
from numpy import random
import sys
import os
import argparse
import itertools
sys.path.append("./")

from interface import evaluate
from optimizer.optimizer import optimize
from models.model_param import ModelParam
from optimizer.aquisition.aquisition_param import AquisitionParam

parser = argparse.ArgumentParser(description='Run Bayesian Optimization')
parser.add_argument('name', help='name of experiment')
parser.add_argument('n_eval', type=int, help='maximum number of iterations', default=10)
parser.add_argument('n_batch', type=int, help='number of parallel evaluations', default=1)
parser.add_argument('parameters', nargs='*', help='tunable parameters')

args = parser.parse_args()
if not os.path.exists(args.name):
    os.makedirs(args.name)
if args.parameters == []:
    args.parameters = ['pipelining', 'pipelined_dma', 'enable_l2', 'cache_queue_size', 'cache_size', 'cache_assoc', 'cache_hit_latency', 'cache_line_sz', 'cache_bandwidth']
    # args.parameters = ['enable_l2', 'tlb_miss_latency', 'tlb_page_size', 'tlb_assoc', 'tlb_bandwidth', 'tlb_max_outstanding_walks']

param_sweeps={
	'cycle_time': range(1, 6),
	'pipelining': [0, 1],
	'enable_l2': [0, 1],
	'pipelined_dma': [0, 1],
	'tlb_entries': range(17),
	'tlb_hit_latency': range(1, 5),
	'tlb_miss_latency': range(10, 21),
	'tlb_page_size': [4096, 8192],
	'tlb_assoc': [4, 8, 16],
	'tlb_bandwidth': [1, 2],
	'tlb_max_outstanding_walks': [4, 8],
        'cache_size': [16384, 32768, 65536, 131072],
        'cache_assoc': [1, 2, 4, 8, 16],
        'cache_hit_latency': range(1, 5),
        'cache_line_sz': [16, 32, 64],
        'cache_queue_size': [32, 64, 128],
        'cache_bandwidth': range(4, 17)
}

grid = np.array(list(itertools.product(*[param_sweeps[p] for p in args.parameters])))
print(grid)

eval_counter = 0

def f(x):
	global eval_counter
	params = {}
	for p, v in zip(args.parameters, x):
		params[p] = v
	try:
            cycle, power, area = evaluate(params, args.name, eval_counter)
        finally:
            eval_counter += 1
	return np.array([cycle, power])    
	

model_params = {}
aquisition_params = {}

frontier, curve, points = optimize(f,
                           grid,
                           ModelParam('gp', model_params),
                           AquisitionParam('smsego', aquisition_params),
                           10,
                           args.n_eval,
                           np.array([2.0, 2.0]),
			   args.n_batch)

np.savetxt('{}/hv.txt'.format(args.name), curve)
np.savetxt('{}/points.txt'.format(args.name), points)
print('Final curve: {0}'.format(curve))
