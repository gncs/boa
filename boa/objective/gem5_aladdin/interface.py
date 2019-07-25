OUTPUT_DIR_KEY = "output_dir"


def create_header_from_template(template_path, file_path, params):
    gem5_options = [
        'cycle_time',
        'pipelining',
        'cache_size',
        'cache_assoc',
        'cache_hit_latency',
        'cache_line_sz',
        'cache_queue_size',
        'cache_bandwith',
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
        'ready_mode',
        'ignore_cache_flush',
    ]

    with open(template_path) as f:
        template = f.read()

    template = template.replace('$OUTPUT_DIR', params[OUTPUT_DIR_KEY])

    for option in [key for key in params if key in gem5_options]:
        template = template.replace('# Insert here', 'set {option} {value}\n# Insert here'.format(option=option,
                                                                                                  value=params[option]))

    with open(file_path, 'w') as f:
        f.write(template)
