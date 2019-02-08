import numpy as np
import sys
import os
import argparse
import itertools
import glob2
sys.path.append("./")

from interface import evaluate
from optimizer.optimizer import optimize
from models.model_param import ModelParam
from optimizer.aquisition.aquisition_param import AquisitionParam

param_sweeps={
	'cycle_time': range(1, 6),
	'pipelining': [0, 1],
	'enable_l2': [0, 1],
	'l2cache_size': [131072, 262144, 524288, 1048576],
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
        'cache_line_sz': [16, 32, 64, 128],
        'cache_queue_size': [32, 64, 128],
        'cache_bandwidth': range(4, 17),
	'ignore_cache_flush': [0, 1],
	'ready_mode': [0, 1]
}


for i in range(5000, 50000):
	params = {}
	for p in param_sweeps:
		params[p] = np.random.choice(param_sweeps[p])
	try:
		evaluate(params, 'collect', i)
		
	except Exception:
		print('failed run')
	finally:
		for file in glob2.glob('./collect/evaluation_{}/**/*'.format(i), recursive=True):
			if os.path.isfile(file) and not (file.endswith('stdout') or file.endswith('0') or file.endswith('fft_tranpose') or file.endswith('outputs')):
				print(file)
				os.remove(file)	

