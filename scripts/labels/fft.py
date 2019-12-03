input_labels = [
    # NB: cache_bandwith is missing!
    'cycle_time',
    'pipelining',
    'cache_size',
    'cache_assoc',
    'cache_hit_latency',
    'cache_line_sz',
    'cache_queue_size',
    'tlb_hit_latency',
    'tlb_miss_latency',
    'tlb_page_size',
    'tlb_entries',
    'tlb_max_outstanding_walks',
    'tlb_assoc',
    'tlb_bandwidth',
    'l2cache_size',
    'enable_l2',
    'pipelined_dma',
    'ignore_cache_flush',
]

output_labels = [
    'cycle',
    'avg_power',
    'idle_fu_cycles',
    'avg_fu_power',
    'avg_fu_dynamic_power',
    'avg_fu_leakage_power',
    'avg_mem_power',
    'avg_mem_dynamic_power',
    'avg_mem_leakage_power',
    'total_area',
    'fu_area',
    'mem_area',
    'num_double_precision_fp_multipliers',
    'num_double_precision_fp_adders',
    'num_trigonometric_units',
    'num_bit-wise_operators_32',
    'num_shifters_32',
    'num_registers_32',
]
