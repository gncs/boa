import os
import shutil
import subprocess

import numpy as np
import tensorflow as tf

from boa.models import GPARModel, FullyFactorizedGPModel, RandomModel
from boa.acquisition.smsego import SMSEGO
from boa.core.utils import InputSpec
from boa.grid import SobolGrid
from boa.objective.gem5_aladdin.util import Gem5DockerRunConfig, run_gem5_simulation_with_configuration, \
    create_gem5_sweep_config

from boa import ROOT_DIR

# Set CPU as available physical device
tf.config.experimental.set_visible_devices([], 'GPU')

config = {
    "run": {
        "experiment_dir": "test_21",
        "seed": 42,
        "num_grid_points": 200000,
        "num_warmup_points": 10,
        "max_num_iterations": 100,
        "model": "gpar",
    },

    "boa": {
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
        InputSpec(name="cycle_time", domain=[1, 2, 3, 4, 5, 6, 10]),

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

        # Maxium number of cache requests can be issued in one cycle
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
        "cycle",
        "avg_power",
        "total_area"
    ],

    "num_targets": 3,
}

gem5_resource_path = os.path.join(ROOT_DIR, "objective/gem5_aladdin/resources")
template_file_path = os.path.join(gem5_resource_path, "machsuite_template.xe")

gem5_copy_items = ["designsweeptypes.py"]

smaug_experiment_dir = "/scratch/gf332/BayesOpt/smaug"
aladdin_experiment_dir = "/scratch/gf332/BayesOpt/gem5-aladdin"


def run_experiment(task):
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    # Create experiment directory
    experiment_dir = os.path.join(smaug_experiment_dir, config["run"]["experiment_dir"])
    docker_experiment_dir = f"/workspace/boa/{config['run']['experiment_dir']}"

    os.makedirs(experiment_dir, exist_ok=True)

    # In an experiment folder, this is the subdirectory where run.sh is located
    run_sh_subdir = f"{task}/{task}/0/"

    # Copy required resources
    for resource_item in gem5_copy_items:
        shutil.copy(src=os.path.join(gem5_resource_path, resource_item),
                    dst=os.path.join(experiment_dir, resource_item))

    # Copy appropriate MachSuite constants
    with open(os.path.join(experiment_dir, "machsuite_constants.xe"), "w") as constants_file:
        result = subprocess.run(
            ["grep",
             task,
             os.path.join(gem5_resource_path, "machsuite_constants.xe")],
            stdout=constants_file,
            text=True,
            check=True,
        )

    grid = SobolGrid(dim_specs=config["inputs"],
                     seed=42,
                     num_grid_points=config["run"]["num_grid_points"],
                     save_dependency_graph_path=os.path.join(experiment_dir, "dependency_graph.png"))

    # -------------------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------------------

    if not os.path.exists(os.path.join(experiment_dir, "warmup_xs.npy")):

        warmup_xs = grid.sample(config["run"]["num_warmup_points"])
        warmup_ys = tf.zeros((0, len(config["outputs"])), dtype=tf.float64)

        # Create warmup sweep configs
        warmup_inputs_settings = grid.input_settings_from_points(points=warmup_xs)

        warmup_configs = [create_gem5_sweep_config(template_file_path=template_file_path,
                                                   output_dir=task,
                                                   input_settings=input_settings)
                          for input_settings in warmup_inputs_settings]

        for warmup_index, warmup_config in enumerate(warmup_configs):
            warmup_dir_name = f"warmup_{warmup_index}"
            warmup_dir = os.path.join(experiment_dir, warmup_dir_name)
            docker_warmup_dir = os.path.join(docker_experiment_dir, warmup_dir_name)

            result_vec = run_gem5_simulation_with_configuration(
                simulation_dir=warmup_dir,
                run_sh_subdir=run_sh_subdir,
                simulation_config=warmup_config,
                output_order=config["outputs"],
                docker_settings=Gem5DockerRunConfig(container_name="smaug",
                                                    simulation_dir=docker_warmup_dir),
            )

            warmup_ys = tf.concat([warmup_ys, result_vec[None, :]], axis=0)

        np.save(os.path.join(experiment_dir, "warmup_xs.npy"), warmup_xs.numpy())
        np.save(os.path.join(experiment_dir, "warmup_ys.npy"), warmup_ys.numpy())

    else:
        warmup_xs = tf.convert_to_tensor(np.load(os.path.join(experiment_dir, "warmup_xs.npy")))
        warmup_ys = tf.convert_to_tensor(np.load(os.path.join(experiment_dir, "warmup_ys.npy")))

    # -------------------------------------------------------------------------
    # Setup for BO
    # -------------------------------------------------------------------------

    surrogate_model = {
        "random": RandomModel(input_dim=len(config["inputs"]),
                              output_dim=len(config["outputs"]),
                              seed=42,
                              num_samples=10,
                              verbose=True),

        "gpar": GPARModel(kernel=config["boa"]["kernel"],
                          input_dim=len(config["inputs"]),
                          output_dim=len(config["outputs"]),
                          verbose=True),

        "ff-gp": FullyFactorizedGPModel(kernel=config["boa"]["kernel"],
                                        input_dim=len(config["inputs"]),
                                        output_dim=len(config["outputs"]),
                                        verbose=True),
    }[config["model"]]

    acq_fun = SMSEGO(**config["boa"]["acq_config"])

    candidate_xs = grid.points

    surrogate_model = surrogate_model.condition_on(xs=warmup_xs,
                                                   ys=warmup_ys)

    surrogate_model.fit(optimizer=config["boa"]["optimizer"],
                        optimizer_restarts=config["boa"]["optimizer_restarts"],
                        iters=config["boa"]["iters"],
                        fit_joint=config["boa"]["fit_joint"],
                        length_scale_init_mode=config["boa"]["length_scale_init"],
                        map_estimate=False,
                        denoising=False,
                        trace=True)

    # -------------------------------------------------------------------------
    # BO loop
    # -------------------------------------------------------------------------

    for eval_index in range(config["run"]["max_num_iterations"]):
        # ---------------------------------------------------------------------
        # Step 1: Evaluate acquisition function on every grid point
        # ---------------------------------------------------------------------
        eval_model = surrogate_model.copy()

        acquisition_values, y_preds = acq_fun.evaluate(model=eval_model,
                                                       xs=eval_model.xs.numpy(),
                                                       ys=eval_model.ys.numpy(),
                                                       candidate_xs=candidate_xs.numpy())

        # ---------------------------------------------------------------------
        # Step 2: Find maximum of acquisition function
        # ---------------------------------------------------------------------
        max_acquisition_index = np.argmax(acquisition_values)
        eval_x = candidate_xs[max_acquisition_index]

        # ---------------------------------------------------------------------
        # Step 3: Run Gem5 simulation at selected location
        # ---------------------------------------------------------------------
        eval_input_settings = grid.input_settings_from_points(eval_x)[0]
        eval_config = create_gem5_sweep_config(template_file_path=template_file_path,
                                               output_dir=task,
                                               input_settings=eval_input_settings)

        eval_dir_name = f"evaluation_{eval_index}"
        eval_dir = os.path.join(experiment_dir, eval_dir_name)
        docker_eval_dir = os.path.join(docker_experiment_dir, eval_dir_name)

        eval_y = run_gem5_simulation_with_configuration(
            simulation_dir=eval_dir,
            run_sh_subdir=run_sh_subdir,
            simulation_config=eval_config,
            output_order=config["outputs"],
            docker_settings=Gem5DockerRunConfig(container_name="smaug",
                                                simulation_dir=docker_eval_dir),
        )

        # ---------------------------------------------------------------------
        # Step 4: Fit surrogate model to dataset augmented by the new datapoint
        # ---------------------------------------------------------------------
        surrogate_model = surrogate_model.condition_on(eval_x, eval_y)

        surrogate_model.fit(optimizer=config["boa"]["optimizer"],
                            optimizer_restarts=config["boa"]["optimizer_restarts"],
                            iters=config["boa"]["iters"],
                            fit_joint=config["boa"]["fit_joint"],
                            length_scale_init_mode=config["boa"]["length_scale_init"],
                            map_estimate=False,
                            denoising=False,
                            trace=True)

        surrogate_model.save(os.path.join(eval_dir, "surrogate_model/model"))

        np.save(os.path.join(eval_dir, "xs.npy"), surrogate_model.xs.numpy())
        np.save(os.path.join(eval_dir, "ys.npy"), surrogate_model.ys.numpy())


if __name__ == "__main__":
    run_experiment(task="fft_transpose")
