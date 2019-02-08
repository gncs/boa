import numpy as np
import re

regexps = [
	re.compile(r'Cycle : (.*)\scycles.*'),
	re.compile(r'Avg Power: (.*)\smW.*'),
	re.compile(r'Idle FU Cycles: (.*)\scycles.*'),
	re.compile(r'Avg FU Power: (.*)\smW.*'),
	re.compile(r'Avg FU Dynamic Power: (.*)\smW.*'),
	re.compile(r'Avg FU leakage Power: (.*)\smW.*'),
	re.compile(r'Avg MEM Power: (.*)\smW.*'),
	re.compile(r'Avg MEM Dynamic Power: (.*)\smW.*'),
	re.compile(r'Avg MEM Leakage Power: (.*)\smW.*'),
	re.compile(r'Total Area: (.*)\suM\^2.*'),
	re.compile(r'FU Area: (.*)\suM\^2.*'),
	re.compile(r'MEM Area: (.*)\suM\^2.*'),
	re.compile(r'Num of Double Precision FP Multipliers: (.*)\s.*'),
	re.compile(r'Num of Double Precision FP Adders: (.*)\s.*'),
	re.compile(r'Num of Trigonometric Units: (.*)\s.*'),
	re.compile(r'Num of Bit-wise Operators \(32-bit\): (.*)\s.*'),
	re.compile(r'Num of Shifters \(32-bit\): (.*)\s.*'),
	re.compile(r'Num of Registers \(32-bit\): (.*)\s.*')
]

header_regexps = [
	re.compile(r'set cycle_time (.*)\s.*'),
	re.compile(r'set pipelining (.*)\s.*'),
	re.compile(r'set cache_size (.*)\s.*'),
	re.compile(r'set cache_assoc (.*)\s.*'),
	re.compile(r'set cache_hit_latency (.*)\s.*'),
	re.compile(r'set cache_line_sz (.*)\s.*'),
	re.compile(r'set cache_queue_size (.*)\s.*'),
	re.compile(r'set tlb_hit_latency (.*)\s.*'),
	re.compile(r'set tlb_miss_latency (.*)\s.*'),
	re.compile(r'set tlb_page_size (.*)\s.*'),
	re.compile(r'set tlb_entries (.*)\s.*'),
	re.compile(r'set tlb_max_outstanding_walks (.*)\s.*'),
	re.compile(r'set tlb_assoc (.*)\s.*'),
	re.compile(r'set tlb_bandwidth (.*)\s.*'),
	re.compile(r'set l2cache_size (.*)\s.*'),
	re.compile(r'set enable_l2 (.*)\s.*'),
	re.compile(r'set pipelined_dma (.*)\s.*'),
	re.compile(r'set ignore_cache_flush (.*)\s.*')
]

def parsefile(filename, regexps):
	in_file = open(filename, "rt") # open file lorem.txt for reading text data
	contents = in_file.read() 
	in_file.close()
	result = []
	for r in regexps:
		m = r.search(contents)
		if m is not None:
			print(m.groups(0))
			result.append(float(m.groups(0)[0]))
		else:
			raise Exception('failed to parse regexp')
	return result
data = []
for i in range(0, 100000):
	try:
		data_line = parsefile('collect/evaluation_{}/fft-transpose/0/outputs/stdout'.format(i), regexps)
		header_line = parsefile('../sweeps/benchmarks/header_collect_{}.xe'.format(i), header_regexps)
		data.append(header_line + data_line)
	except Exception as e: print(e)

np.savetxt('fft_dataset.csv', np.array(data))



