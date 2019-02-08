import os
import sys
import re
from shutil import copyfile
from optimizer.optimizer_helper import InvalidParameters
sys.path.append('/workspace/gem5-aladdin/sweeps')

TASK_NAME='fft-transpose'

def evaluate(params, dir, counter):

    create_header_from_template(params,
                                '/workspace/gem5-aladdin/sweeps/benchmarks/header_{0}_{1}.xe'.format(dir, counter),
                                '{0}/evaluation_{1}'.format(dir, counter))
    os.system('cp -r /workspace/gem5-aladdin/bo_script/traces {0}/evaluation_{1}/'.format(dir, counter))
    os.chdir('/workspace/gem5-aladdin/sweeps')
    os.system('python generate_design_sweeps.py benchmarks/header_{0}_{1}.xe'.format(dir, counter))
    os.chdir('/workspace/gem5-aladdin/bo_script/{0}/evaluation_{1}/{2}/0/'.format(dir, counter, TASK_NAME))
    os.system('sh run.sh'.format(dir, counter, TASK_NAME))
    os.chdir('/workspace/gem5-aladdin/bo_script')
    
    try:
        return get_cycle_power_area('{0}/evaluation_{1}/{2}/0/outputs/stdout'.format(dir, counter, TASK_NAME))
    except:
        if not os.path.exists('{0}/errors'.format(dir)):
            os.makedirs('{0}/errors'.format(dir))
        copyfile('{0}/evaluation_{1}/{2}/0/outputs/stderr'.format(dir, counter, TASK_NAME), '{0}/errors/stderr_{1}'.format(dir, counter))
        raise InvalidParameters

def get_cycle_power_area(output_file):
    cycle = [re.findall(r'Cycle : (.*) cycles',line) for line in open(output_file)]
    cycle = [c for l in cycle for c in l]
    power = [re.findall(r'Avg Power: (.*) mW',line) for line in open(output_file)]
    power = [p for l in power for p in l]
    area = [re.findall(r'Total Area: (.*) uM',line) for line in open(output_file)]
    area = [a for l in area for a in l]

    return [int(cycle[0]), float(power[0]), float(area[0])]


def create_header_from_template(params,
                                header_name,
                                output_dir,
                                template_name='template.xe'):

    keys = ['cycle_time',
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
            'ready_mode from',
            'ignore_cache_flush']

    # Create the header file for the file that will be modeled
    with open(template_name, 'r') as f:
        h = f.read()

    h = h.replace('$OUTPUT_DIR', '/workspace/gem5-aladdin/bo_script/{0}'.format(output_dir))

    for key in keys:
        if key in params:
            h = h.replace('# Insert here',
                          'set {0} {1}\n# Insert here'.format(key, params[key]))

    with open(header_name, 'w') as f:
        f.write(h)


