import re
from typing import List, Dict

output_regexps = {
    'cycle': re.compile(r'Cycle : (.*) cycles'),
    'avg_power': re.compile(r'Avg Power: (.*) mW'),
    'idle_fu_cycles': re.compile(r'Idle FU Cycles: (.*) cycles'),
    'avg_fu_cycles': re.compile(r'Avg FU Power: (.*) mW'),
    'avg_fu_dynamic_power': re.compile(r'Avg FU Dynamic Power: (.*) mW'),
    'avg_fu_leakage_power': re.compile(r'Avg FU leakage Power: (.*) mW'),
    'avg_mem_power': re.compile(r'Avg MEM Power: (.*) mW'),
    'avg_mem_dynamic_power': re.compile(r'Avg MEM Dynamic Power: (.*) mW'),
    'avg_mem_leakage_power': re.compile(r'Avg MEM Leakage Power: (.*) mW'),
    'total_area': re.compile(r'Total Area: (.*) uM\^2'),
    'fu_area': re.compile(r'FU Area: (.*) uM\^2'),
    'mem_area': re.compile(r'MEM Area: (.*) uM\^2'),
    'num_double_precision_fp_multipliers': re.compile(r'Num of Double Precision FP Multipliers: (.*)\s'),
    'num_double_precision_fp_adders': re.compile(r'Num of Double Precision FP Adders: (.*)'),
    'num_trigonometric_units': re.compile(r'Num of Trigonometric Units: (.*)'),
    'num_bit-wise_operators_32': re.compile(r'Num of Bit-wise Operators \(32-bit\): (.*)'),
    'num_shifters_32': re.compile(r'Num of Shifters \(32-bit\): (.*)'),
    'num_registers_32': re.compile(r'Num of Registers \(32-bit\): (.*)'),
}

header_regexps = {
    'cycle_time': re.compile(r'set cycle_time (.*)\s.*'),
    'pipelining': re.compile(r'set pipelining (.*)\s.*'),
    'cache_size': re.compile(r'set cache_size (.*)\s.*'),
    'cache_assoc': re.compile(r'set cache_assoc (.*)\s.*'),
    'cache_hit_latency': re.compile(r'set cache_hit_latency (.*)\s.*'),
    'cache_line_sz': re.compile(r'set cache_line_sz (.*)\s.*'),
    'cache_queue_size': re.compile(r'set cache_queue_size (.*)\s.*'),
    'tlb_hit_latency': re.compile(r'set tlb_hit_latency (.*)\s.*'),
    'tlb_miss_latency': re.compile(r'set tlb_miss_latency (.*)\s.*'),
    'tlb_page_size': re.compile(r'set tlb_page_size (.*)\s.*'),
    'tlb_entries': re.compile(r'set tlb_entries (.*)\s.*'),
    'tlb_max_outstanding_walks': re.compile(r'set tlb_max_outstanding_walks (.*)\s.*'),
    'tlb_assoc': re.compile(r'set tlb_assoc (.*)\s.*'),
    'tlb_bandwidth': re.compile(r'set tlb_bandwidth (.*)\s.*'),
    'l2cache_size': re.compile(r'set l2cache_size (.*)\s.*'),
    'enable_l2': re.compile(r'set enable_l2 (.*)\s.*'),
    'pipelined_dma': re.compile(r'set pipelined_dma (.*)\s.*'),
    'ignore_cache_flush': re.compile(r'set ignore_cache_flush (.*)\s.*'),
}


def get_cycle_power_area(file_path: str) -> List[float]:
    results = parse_file(file_path=file_path, regexps=output_regexps)

    return [results['cycle'], results['avg_power'], results['total_area']]


def parse_file(file_path: str, regexps: dict) -> Dict[str, float]:
    with open(file_path) as f:
        contents = f.read()

    results = {}
    for key, regexp in regexps.items():
        match = regexp.search(contents).group(1)

        if match:
            results[key] = float(match)

        else:
            raise ValueError(f"No match found for {key}!")

    return results
