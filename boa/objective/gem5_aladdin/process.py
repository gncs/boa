import re

output_regexps = [
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


def parse_file(filename, regexps):
    with open(filename) as f:
        contents = f.read()

    results = []
    for regexp in regexps:
        try:
            m = regexp.search(contents)
            results.append(float(m.groups(0)[0]))
        except Exception:
            raise Exception("Failed to parse regular expression '" + regexp.pattern + "'")

    return results
