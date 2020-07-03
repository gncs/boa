"""
import boa

define config

create folders
    - experiment folder
        - log folder
        - model folder
        - data folder

for 1..N:

    run query

    save datapoints

"""
from collections import OrderedDict

from string import Template

import abc
import os
import shutil
import subprocess
import datetime
import json

from typing import Dict, NamedTuple, Tuple

import numpy as np
import tensorflow as tf

from boa.objective.gem5_aladdin import parse_file, output_regexps
from boa.models import GPARModel
from boa.acquisition.smsego import SMSEGO
from boa.core.utils import NumpyEncoder, InputSpec
from boa.grid import SobolGrid

from boa import ROOT_DIR


class Gem5DockerRunConfig(NamedTuple):
    container_name: str
    simulation_dir: str


config = {
    "run": {
        "experiment_dir": "test_2",
        "seed": 42,
        "num_grid_points": 2000,
        "num_warmup_points": 2,
        "max_num_iterations": 10,
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
            "reference": np.array([0., 0., 0.]),
            "output_slice": (-3, None)
        }
    },
    "inputs": [
        # Cycle time swept from 1 to 5ns.
        InputSpec(name="cycle_time", domain=tf.range(1, 6)),

        # With or without pipelining.
        InputSpec(name="pipelining", domain=[0, 1]),

        # Cache size swept from 16kB to 128kB
        # Note: cache_size has to be x * cache_line_sz * cache_assoc
        InputSpec(name="cache_size", domain=2**tf.range(14, 18)),

        # Cache associativity
        # Note: This setting only seems to support powers of two
        InputSpec(name="cache_assoc", domain=2**tf.range(0, 5)),

        InputSpec(name="cache_hit_latency", domain=tf.range(1, 5)),

        # Note: This setting only seems to support powers of two
        #InputSpec(name="cache_line_sz", domain=2 ** tf.range(4, 8)),
        InputSpec(name="cache_line_sz", domain=64),

        # Internal cache queue size
        # Note: This setting only seems to support powers of two
        InputSpec(name="cache_queue_size", domain=2 ** tf.range(5, 8)),

        # Maxium number of cache requests can be issued in one cycle
        InputSpec(name="cache_bandwidth", domain=tf.range(4, 17)),
        InputSpec(name="tlb_hit_latency", domain=tf.range(1, 5)),
        InputSpec(name="tlb_miss_latency", domain=tf.range(10, 21)),

        #InputSpec(name="tlb_page_size", domain=2**tf.range(12, 14)),
        InputSpec(name="tlb_page_size", domain=4096),

        # 0 means infinite TLB entries
        # Note: tlb_entries has to be x * tlb_assoc
        InputSpec(name="tlb_entries", domain=tf.range(0, 17)),
        InputSpec(name="tlb_max_outstanding_walks", domain=2**tf.range(2, 4)),
        InputSpec(name="tlb_assoc", domain=tf.range(4, 17)),
        InputSpec(name="tlb_bandwidth", domain=[1, 2]),

        # Shared L2 size form 128kB to 1MB
        InputSpec(name="l2cache_size", domain=2**tf.range(17, 21)),

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


def create_gem5_sweep_config(template_file_path: str,
                             output_dir: str,
                             input_settings: Dict,
                             source_dir: str = "/workspace/gem5-aladdin/src/aladdin/MachSuite",
                             simulator: str = "gem5-cpu",
                             generation_commands: Tuple[str] = ("configs", "trace", "gem5_binary")):
    # Creates commands of the form "generate x" on multiple lines
    generation_commands = '\n'.join(map(lambda x: "generate " + x, generation_commands))

    # Creates the commands describing what inputs to try next
    eval_settings = [f"set {inp} {val}" for inp, val in input_settings.items()]
    eval_settings = '\n'.join(eval_settings)

    template_file = open(template_file_path)

    template = Template(template_file.read())

    sweep_config = template.substitute({"task": "fft_transpose",
                                        "generation_commands": generation_commands,
                                        "output_dir": output_dir,
                                        "source_dir": source_dir,
                                        "simulator": simulator,
                                        "evaluation_settings": eval_settings})

    return sweep_config


def create_experiment_dirs(experiment_path,
                           log_dir_name="logs",
                           data_dir_name="data",
                           model_dir_name="models"):
    log_path = os.path.join(experiment_path, log_dir_name)
    data_path = os.path.join(experiment_path, data_dir_name)
    model_path = os.path.join(experiment_path, model_dir_name)

    try:

        os.makedirs(experiment_path, exist_ok=False)

        for dir_name in [log_path, data_path, model_path]:
            os.makedirs(dir_name)

    except FileExistsError:
        raise FileExistsError(f"Folder already exists: {experiment_path}!")

    paths = {
        "log": log_path,
        "data": data_path,
        "model": model_path,
    }

    return paths


def get_output_vector_from_dict(output_dict):
    output_list = [output_dict[output_name] for output_name in config["outputs"]]
    output_vec = tf.convert_to_tensor(output_list)
    output_vec = tf.cast(output_vec, tf.float64)

    return output_vec


def run_gem5_simulation_with_configuration(simulation_dir: str,
                                           run_sh_subdir: str,
                                           simulation_config: str,
                                           docker_settings: Gem5DockerRunConfig = None,
                                           template_name: str = "template.xe",
                                           template_generation_out_file_name: str = "template_generation_output.txt",
                                           generate_design_sweeps_py_path: str = "/workspace/gem5-aladdin/sweeps/generate_design_sweeps.py",
                                           simulation_output_file_name: str = "simulation_output.txt",
                                           stdout_path: str = "outputs/stdout",
                                           stderr_path: str = "outputs/stderr",
                                           results_file_name: str = "results.json",
                                           ):
    if docker_settings is not None and not isinstance(docker_settings, Gem5DockerRunConfig):
        raise TypeError(f"docker_settings must be of type Gem5DockerRunConfig, but had type {type(docker_settings)}!")

    os.makedirs(simulation_dir, exist_ok=True)

    # Write template
    with open(os.path.join(simulation_dir, template_name), "w") as template_file:
        template_file.write(simulation_config)

    # Generate design sweep
    print(f"Generating template file in {simulation_dir}!")
    with open(os.path.join(simulation_dir, template_generation_out_file_name), "w") as log_file:

        if docker_settings is None:
            subprocess_commands = [
                "/workspace/gem5-aladdin/sweeps/generate_design_sweeps.py",
                template_name
            ]

        else:
            subprocess_commands = [
                "docker",
                "exec",
                "-w",
                docker_settings.simulation_dir,
                docker_settings.container_name,
                generate_design_sweeps_py_path,
                template_name]

        result = subprocess.run(
            subprocess_commands,
            stdout=log_file,
            stderr=log_file,
            text=True,
            check=True,
        )

    # Run warmup simulation
    if docker_settings is None:
        run_script_dir = os.path.join(simulation_dir, run_sh_subdir)
    else:
        run_script_dir = os.path.join(docker_settings.simulation_dir, run_sh_subdir)
    print(f"Running simulation in {run_script_dir}!")

    with open(os.path.join(simulation_dir, simulation_output_file_name), "w") as log_file:

        if docker_settings is None:
            pass

        else:
            subprocess_commands = [
                "docker",
                "exec",
                "-w",
                run_script_dir,
                docker_settings.container_name,
                "bash",
                "run.sh"
            ]

        try:
            result = subprocess.run(
                subprocess_commands,
                stdout=log_file,
                stderr=log_file,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            stderr_path = os.path.join(simulation_dir, run_sh_subdir, stderr_path)
            log_file.writelines(["========================\n",
                                 f"Copied from {stderr_path}\n"
                                 "========================\n", ])

            with open(stderr_path) as stderr_log_file:
                stderr_log = stderr_log_file.read()

                log_file.write(stderr_log)

            print(str(e))
            raise ValueError(stderr_log)

    results = parse_file(os.path.join(simulation_dir, run_sh_subdir, stdout_path),
                         output_regexps)

    with open(os.path.join(simulation_dir, results_file_name), "w") as results_file:
        json.dump(results, results_file)

    result_vec = get_output_vector_from_dict(results)

    return result_vec


def run_experiment(task,
                   log_dir_name="logs",
                   data_dir_name="data",
                   model_dir_name="models",
                   ):
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    # Create experiment directory
    experiment_dir = os.path.join(smaug_experiment_dir, config["run"]["experiment_dir"])
    docker_experiment_dir = f"/workspace/boa/{config['run']['experiment_dir']}"

    os.makedirs(experiment_dir)

    # In an experiment folder, this is the subdirector where run.sh is located
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

    grid = SobolGrid(dim_spec=config["inputs"],
                     seed=42,
                     num_grid_points=config["run"]["num_grid_points"])

    # -------------------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------------------

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
            docker_settings=Gem5DockerRunConfig(container_name="smaug",
                                                simulation_dir=docker_warmup_dir),
        )

        warmup_ys = tf.concat([warmup_ys, result_vec[None, :]], axis=0)

    # -------------------------------------------------------------------------
    # Setup for BO
    # -------------------------------------------------------------------------
    surrogate_model = GPARModel(kernel=config["boa"]["kernel"],
                                input_dim=len(config["inputs"]),
                                output_dim=len(config["outputs"]),
                                verbose=True)

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

    xs = warmup_xs
    ys = warmup_ys

    for eval_index in range(config["run"]["max_num_iterations"]):
        eval_model = surrogate_model.copy()

        acquisition_values, y_preds = acq_fun.evaluate(model=eval_model,
                                                       denoising=False,
                                                       xs=xs,
                                                       ys=ys,
                                                       candidate_xs=candidate_xs)
        max_acquisition_index = np.argmax(acquisition_values)
        eval_x = candidate_xs[max_acquisition_index]
        y_pred = y_preds[max_acquisition_index]

        # Gem5 evaluation
        eval_input_settings = grid.input_settings_from_points(eval_x)[0]
        eval_config = create_gem5_sweep_config(template_file_path=template_file_path,
                                               output_dir=task,
                                               input_settings=eval_input_settings)

        eval_dir_name = f"evaluation_{eval_index}"
        eval_dir = os.path.join(experiment_dir, eval_dir_name)
        docker_eval_dir = os.path.join(docker_experiment_dir, eval_dir)

        eval_y = run_gem5_simulation_with_configuration(
            simulation_dir=eval_dir,
            run_sh_subdir=run_sh_subdir,
            simulation_config=eval_config,
            docker_settings=Gem5DockerRunConfig(container_name="smaug",
                                                simulation_dir=docker_eval_dir),
        )

        surrogate_model.condition_on(eval_x, eval_y)

        surrogate_model.fit(optimizer=config["boa"]["optimizer"],
                            optimizer_restarts=config["boa"]["optimizer_restarts"],
                            iters=config["boa"]["iters"],
                            fit_joint=config["boa"]["fit_joint"],
                            length_scale_init_mode=config["boa"]["length_scale_init"],
                            map_estimate=False,
                            denoising=False,
                            trace=True)

    print(xs, ys)


if __name__ == "__main__":
    run_experiment(task="fft_transpose")
