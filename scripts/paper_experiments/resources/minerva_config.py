import tensorflow as tf
from boa.core.utils import InputSpec

config = {
    "run": {
        "experiment_dir": "test_bnn_out_std",
        "seed": 42,
        "num_grid_points": 20000,
        "num_warmup_points": 10,
        "max_num_iterations": 50,
    },

    "boa": {
        "model": "bnn",
        "optimizer": "l-bfgs-b",
        "optimizer_restarts": 3,
        "iters": 1000,
        "fit_joint": False,
        "length_scale_init": "l2_median",
        "kernel": "matern52",
        "acq_config": {
            "gain": 1.,
            "epsilon": 0.01,
            "output_slice": (-3, None)
        }
    },
    "inputs": [
        # the following parameters should be power of 2:
        # l2cache_size, tlb_assoc, tlb_entries, cache_line_sz, cache_assoc, cache_size

        # Cycle time swept from 1 to 5ns.
        InputSpec(name="cycle_time", domain=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),

        # With or without pipelining.
        InputSpec(name="pipelining", domain=[0, 1]),

        # Minimum: 8192 Bytes
        # Cache size swept from 16kB to 128kB
        # Note: cache_size has to be x * cache_line_sz * cache_assoc
        # Note: the 64 factor comes from cache_line_sz
        InputSpec(name="cache_size", domain=2 ** tf.range(5, 9),
                  formula=lambda x, cache_assoc: x * cache_assoc * 64),

        # Cache associativity
        # Note: This setting only seems to support powers of two
        InputSpec(name="cache_assoc", domain=2 ** tf.range(2, 5)),
        # InputSpec(name="cache_assoc", domain=tf.range(0, 5), formula=lambda x: 2**x),

        InputSpec(name="cache_hit_latency", domain=tf.range(1, 5)),

        # Note: This setting only seems to support powers of two
        # InputSpec(name="cache_line_sz", domain=2 ** tf.range(4, 8)),
        # InputSpec(name="cache_line_sz", domain=[64]),

        # Internal cache queue size
        # Note: This setting only seems to support powers of two
        InputSpec(name="cache_queue_size", domain=2 ** tf.range(5, 8)),

        # Maxim um number of cache requests can be issued in one cycle
        InputSpec(name="cache_bandwidth", domain=tf.range(4, 17)),
        InputSpec(name="tlb_hit_latency", domain=tf.range(1, 5)),
        InputSpec(name="tlb_miss_latency", domain=tf.range(10, 21)),

        # InputSpec(name="tlb_page_size", domain=2**tf.range(12, 14)),
        # InputSpec(name="tlb_page_size", domain=[4096]),

        # Note: tlb_entries has to be x * tlb_assoc
        InputSpec(name="tlb_entries", domain=2 ** tf.range(0, 3),
                  formula=lambda x, tlb_assoc: x * tlb_assoc),

        InputSpec(name="tlb_max_outstanding_walks", domain=[4, 8]),
        InputSpec(name="tlb_assoc", domain=2 ** tf.range(2, 5)),
        InputSpec(name="tlb_bandwidth", domain=[1, 2]),

        # Shared L2 size form 128kB to 1MB
        InputSpec(name="l2cache_size", domain=2 ** tf.range(17, 21)),

        # Use L2 or not
        InputSpec(name="enable_l2", domain=[0, 1]),

        # Use pipelined DMA optimization
        InputSpec(name="pipelined_dma", domain=[0, 1]),

        # Ready bits optimization optimization
        InputSpec(name="ready_mode", domain=[0, 1]),

        # Whether to ignore the DMA induced cache flush overhead
        InputSpec(name="ignore_cache_flush", domain=[0, 1]),
    ],

    "outputs": [
        'idle_fu_cycles',
        'avg_fu_cycles',
        'avg_fu_dynamic_power',
        'avg_fu_leakage_power',
        'avg_mem_power',
        'avg_mem_dynamic_power',
        'avg_mem_leakage_power',
        'fu_area',
        'mem_area',
        "cycle",
        "avg_power",
        "total_area",
    ],

    "num_targets": 3,
}
