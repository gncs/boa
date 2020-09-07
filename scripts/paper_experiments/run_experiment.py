from typing import Dict, Tuple, NamedTuple, List

import os
import fcntl
import shutil
import subprocess
import time 

import json
import numpy as np
import tensorflow as tf

from boa.models import GPARModel, FullyFactorizedGPModel, RandomModel
from boa.models.bnn.multi_output_bnn import BasicBNN
from boa.acquisition.smsego import SMSEGO
from boa.grid import SobolGrid
from boa.objective.gem5_aladdin.util import create_gem5_sweep_config, get_output_vector_from_dict
from boa.objective.gem5_aladdin.process import parse_file, output_regexps

# Note: run_experiment.py should always be run in a directory that contains a valid MachSuite or SMAUG config.py file.
try:
    from config import config
except (ModuleNotFoundError, ImportError):
    print("You must run this script where a valid MachSuite or SMAUG config.py file is located!")
    raise

# Set CPU as available physical device
tf.config.experimental.set_visible_devices([], 'GPU')

machsuite_tasks = [
    "fft_transpose",
    "stencil3d",
    "gemm"
]

smaug_tasks = [
    "minerva"
]


def train_surrogate_model(surrogate_model, model_config, model):

    if model == "sgs":
        # Random model neds no fitting
        pass

    elif model == "bnn":
        surrogate_model.fit(num_samples=model_config["num_samples"],
                            burnin=model_config["burnin"],
                            keep_every=model_config["keep_every"])

    elif model in ["ff-gp", "gpar-targets", "gpar-full"]:
        surrogate_model.fit(optimizer=model_config["optimizer"],
                            optimizer_restarts=model_config["optimizer_restarts"],
                            iters=model_config["iters"],
                            fit_joint=model_config["fit_joint"],
                            length_scale_init_mode=model_config["length_scale_init"],
                            map_estimate=False,
                            denoising=False,
                            trace=True)

    else:
        raise NotImplementedError



def pareto_frontier(points):
    points = points.numpy()

    is_pareto = np.ones(points.shape[0], dtype=bool)

    for i, point in enumerate(points):

        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(points[is_pareto, :] < point, axis=1)
            is_pareto[i] = True

    return tf.convert_to_tensor(is_pareto)


def run_machsuite_simulation_with_configuration(task: str,
                                                simulation_dir: str,
                                                run_sh_subdir: str,
                                                simulation_config: str,
                                                output_order: List[str],
                                                singularity_image: str,
                                                template_name: str = "template.xe",
                                                template_generation_out_file_name: str = "template_generation_output.txt",
                                                generate_design_sweeps_py_path: str = "/workspace/gem5-aladdin/sweeps/generate_design_sweeps.py",
                                                simulation_output_file_name: str = "simulation_output.txt",
                                                lockfile_path: str = "/rds/user/gf332/hpc-work/BayesOpt/experiments/singularity.lock",
                                                stdout_path: str = "outputs/stdout",
                                                stderr_path: str = "outputs/stderr",
                                                results_file_name: str = "results.json",
                                                ):
    os.makedirs(simulation_dir, exist_ok=True)

    run_script_dir = os.path.join(simulation_dir, run_sh_subdir)
    nonaccel_file_path = os.path.join(run_script_dir, f"{task}-gem5")
    accel_file_path = os.path.join(run_script_dir, f"{task}-gem5-accel")

    # Write template
    with open(os.path.join(simulation_dir, template_name), "w") as template_file:
        template_file.write(simulation_config)

    # Generate design sweep

    # Make sure this process is the only one trying to use the write access to singularity
    wait_start_time = time.monotonic()

    lockfile = open(lockfile_path, "w")
    fcntl.lockf(lockfile, fcntl.LOCK_EX)
    lockfile.write(str(os.getpid()))
    lockfile.flush()

    wait_time = time.monotonic() - wait_start_time
    readable_wait_time = time.strftime("%H hours %M minutes %S seconds", time.gmtime(wait_time))

    print(f"Lock obtained by PID {os.getpid()}. Waited {readable_wait_time} for design generation.")

    print(f"Generating template file in {simulation_dir}!")
    with open(os.path.join(simulation_dir, template_generation_out_file_name), "w") as log_file:

        result = subprocess.run([
            "singularity",
            "exec",
            "-B",
            "/rds:/rds",
            "-w",
            "--pwd",  # Changes the working directory to the appropriate one
            simulation_dir,
            singularity_image,
            generate_design_sweeps_py_path,
            template_name],
            stdout=log_file,
            stderr=log_file,
            check=True,
        )

        nonaccel_actual_path = os.readlink(nonaccel_file_path)
        accel_actual_path = os.readlink(accel_file_path)

        os.unlink(nonaccel_file_path)
        os.unlink(accel_file_path)

        # Copy the compiled binaries
        result = subprocess.run([
            "singularity",
            "exec",
            singularity_image,
            "cp",
            nonaccel_actual_path,
            nonaccel_file_path],
            stdout=log_file,
            stderr=log_file,
            check=True,
        )

        result = subprocess.run([
            "singularity",
            "exec",
            singularity_image,
            "cp",
            accel_actual_path,
            accel_file_path],
            stdout=log_file,
            stderr=log_file,
            check=True,
        )

    # Release lock by closeing the lockfile
    lockfile.close()

    print(f"Running simulation in {run_script_dir}!")

    with open(os.path.join(simulation_dir, simulation_output_file_name), "w") as log_file:
        try:
            result = subprocess.run([
                "singularity",
                "exec",
                "--pwd",
                run_script_dir,
                singularity_image,
                "bash",
                "run.sh"
            ],
                stdout=log_file,
                stderr=log_file,
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

    os.remove(nonaccel_file_path)
    os.remove(accel_file_path)

    with open(os.path.join(simulation_dir, results_file_name), "w") as results_file:
        json.dump(results, results_file)

    result_vec = get_output_vector_from_dict(results, output_order=output_order)

    return result_vec


def run_experiment():
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    seed = config["run"]["seed"]
    experiment_dir = config["run"]["experiment_dir"]
    task = config["run"]["task"]
    model = config["run"]["model"]

    warmup_dir = config["run"]["warmup_base_dir"]
    num_warmup_points = config['run']['num_warmup_points']

    machsuite_template_path = os.path.join(experiment_dir, "machsuite_template.xe")

    tf.random.set_seed(seed)

    grid = SobolGrid(dim_specs=config["inputs"],
                     seed=config["run"]["seed"],
                     num_grid_points=config["run"]["num_grid_points"],
                     save_dependency_graph_path=os.path.join(experiment_dir, "dependency_graph.png"))

    # In an experiment folder, this is the subdirectory where run.sh is located
    run_sh_subdir = f"{task}/{task}/0/"

    # -------------------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------------------
    warmup_xs_path = os.path.join(warmup_dir, f"warmup_xs_{num_warmup_points}.npy")
    warmup_ys_path = os.path.join(warmup_dir, f"warmup_ys_{num_warmup_points}.npy")

    if not os.path.exists(warmup_xs_path):

        print("Necessary warmup points don't exist yet, creating them now...")

        warmup_xs = grid.sample(num_warmup_points)
        warmup_ys = tf.zeros((0, len(config["outputs"])), dtype=tf.float64)

        # Create warmup sweep configs
        warmup_inputs_settings = grid.input_settings_from_points(points=warmup_xs)

        # Generate warmups
        for warmup_index, input_settings in enumerate(warmup_inputs_settings):

            warmup_dir_name = f"warmup_{warmup_index}"
            current_warmup_dir = os.path.join(experiment_dir, warmup_dir_name)

            current_warmup_xs_path = os.path.join(warmup_dir, f"warmup_xs_{warmup_index + 1}.npy")
            current_warmup_ys_path = os.path.join(warmup_dir, f"warmup_ys_{warmup_index + 1}.npy")

            if os.path.exists(current_warmup_ys_path):
                print(f"{current_warmup_dir}, already exists, skipping!")
                warmup_ys = tf.convert_to_tensor(np.load(current_warmup_ys_path))
                continue

            warmup_config = create_gem5_sweep_config(template_file_path=machsuite_template_path,
                                                     output_dir=os.path.join(current_warmup_dir, task),
                                                     input_settings=input_settings,
                                                     generation_commands=("configs",))

            result_vec = run_machsuite_simulation_with_configuration(
                task=task,
                simulation_dir=current_warmup_dir,
                run_sh_subdir=run_sh_subdir,
                simulation_config=warmup_config,
                output_order=config["outputs"],
                singularity_image=config["run"]["singularity_image"],
            )

            warmup_ys = tf.concat([warmup_ys, result_vec[None, :]], axis=0)

            np.save(current_warmup_xs_path, warmup_xs.numpy())
            np.save(current_warmup_ys_path, warmup_ys.numpy())

            # Remove the simulation files after all important information has been extracted.
            # This is very important, because the simulation directory is effectively useless after,
            # can be regenerated easily and takes up a lot of space.
            shutil.rmtree(os.path.join(current_warmup_dir, task))

    else:
        print("Using precomputed warmup points!")
        warmup_xs = tf.convert_to_tensor(np.load(warmup_xs_path))
        warmup_ys = tf.convert_to_tensor(np.load(warmup_ys_path))

    # -------------------------------------------------------------------------
    # Setup for BO
    # -------------------------------------------------------------------------

    # Reset the random seed, because depending on whether warmup points were generated or not,
    # we might get different results here
    tf.random.set_seed(seed)

    # Truncate the datapoints if necessary
    if model in ["ff-gp", "gpar-targets"]:
        warmup_ys = warmup_ys[:, -config["num_targets"]:]

    model_config = config["models"][model]

    if model == "sgs":
        surrogate_model = RandomModel(input_dim=len(config["inputs"]),
                                      output_dim=len(config["outputs"]),
                                      seed=seed,
                                      num_samples=10,
                                      verbose=True)
    elif model == "gpar-targets":
        surrogate_model = GPARModel(kernel=model_config["kernel"],
                                    input_dim=len(config["inputs"]),
                                    output_dim=config["num_targets"],
                                    verbose=True)

    elif model == "gpar-full":
        surrogate_model = GPARModel(kernel=model_config["kernel"],
                                    input_dim=len(config["inputs"]),
                                    output_dim=len(config["outputs"]),
                                    verbose=True)

    elif model == "ff-gp":
        surrogate_model = FullyFactorizedGPModel(kernel=model_config["kernel"],
                                                 input_dim=len(config["inputs"]),
                                                 output_dim=config["num_targets"],
                                                 verbose=True)

    elif model == "bnn":
        surrogate_model = BasicBNN(input_dim=len(config["inputs"]),
                                   output_dim=len(config["outputs"]))

    else:
        raise NotImplementedError

    acq_fun = SMSEGO(**config["run"]["acq_config"])

    surrogate_model = surrogate_model.condition_on(xs=warmup_xs,
                                                   ys=warmup_ys)

    if not os.path.exists(os.path.join(experiment_dir, "evaluation_0", "xs.npy")):
        train_surrogate_model(surrogate_model, model_config, config["run"]["model"])

    # -------------------------------------------------------------------------
    # BO loop
    # -------------------------------------------------------------------------

    for eval_index in range(config["run"]["max_num_evaluations"]):

        # Make sure every evaluation has its own unique seed
        tf.random.set_seed(seed + eval_index + 1)

        eval_dir_name = f"evaluation_{eval_index}"
        eval_dir = os.path.join(experiment_dir, eval_dir_name)

        grid_save_path = os.path.join(eval_dir, "grid.npy")

        xs_save_path = os.path.join(eval_dir, "xs.npy")
        ys_save_path = os.path.join(eval_dir, "ys.npy")

        if os.path.exists(ys_save_path):

            print(f"Evaluation {eval_index} at: {ys_save_path} already exists, skipping!")

            # If the grid is saved in this evaluation directory, it is the latest one.
            if os.path.exists(grid_save_path):
                print("Last saved model found")

                grid.points = tf.convert_to_tensor(np.load(grid_save_path))
                loaded_xs = tf.convert_to_tensor(np.load(xs_save_path))
                loaded_ys = tf.convert_to_tensor(np.load(ys_save_path))

                surrogate_model = surrogate_model.condition_on(loaded_xs, loaded_ys, keep_previous=False)

                tf.random.set_seed(seed + eval_index + 1)
                train_surrogate_model(surrogate_model, model_config, config["run"]["model"])

            continue

        print(f"{grid.points.shape[0]} points in the grid at iteration {eval_index + 1}")
        # ---------------------------------------------------------------------
        # Step 1: Evaluate acquisition function on every grid point
        # ---------------------------------------------------------------------
        acquisition_values, y_preds = acq_fun.evaluate(model=surrogate_model,
                                                       xs=surrogate_model.xs.numpy(),
                                                       ys=surrogate_model.ys.numpy(),
                                                       candidate_xs=grid.points.numpy())

        # ---------------------------------------------------------------------
        # Step 2: Find maximum of acquisition function
        # ---------------------------------------------------------------------
        pf_indices = pareto_frontier(tf.convert_to_tensor(y_preds[:, -config["num_targets"]:]))

        num_pareto_points = tf.reduce_sum(tf.cast(pf_indices, tf.int32))
        num_extra_samples_per_point = 1000 // num_pareto_points + 1

        print(
            f"{num_pareto_points} Pareto optimal points in grid, adding {num_extra_samples_per_point} samples around each.")

        pareto_xs = grid.points[pf_indices]

        # Try some extra points around the best solutions
        for pareto_x in pareto_xs:
            grid.sample_grid_around_point(pareto_x, num_samples=num_extra_samples_per_point, add_to_grid=True)

        acquisition_values_, y_preds_ = acq_fun.evaluate(model=surrogate_model,
                                                         xs=surrogate_model.xs.numpy(),
                                                         ys=surrogate_model.ys.numpy(),
                                                         candidate_xs=grid.points.numpy())

        max_acquisition_index_ = np.argmax(acquisition_values_)
        eval_x = grid.points[max_acquisition_index_]

        # ---------------------------------------------------------------------
        # Step 3: Run simulation at selected location
        # ---------------------------------------------------------------------
        if task in machsuite_tasks:
            eval_input_settings = grid.input_settings_from_points(eval_x)[0]
            eval_config = create_gem5_sweep_config(template_file_path=machsuite_template_path,
                                                   output_dir=task,
                                                   input_settings=eval_input_settings,
                                                   generation_commands=("configs",))

            eval_y = run_machsuite_simulation_with_configuration(
                task=task,
                simulation_dir=eval_dir,
                run_sh_subdir=run_sh_subdir,
                simulation_config=eval_config,
                output_order=config["outputs"],
                singularity_image=config["run"]["singularity_image"],
            )

            if model in ["ff-gp", "gpar-targets"]:
                eval_y = eval_y[-config["num_targets"]:]

        elif task in smaug_tasks:
            raise NotImplementedError

        else:
            raise NotImplementedError

        # ---------------------------------------------------------------------
        # Step 4: Fit surrogate model to dataset augmented by the new datapoint
        # ---------------------------------------------------------------------
        surrogate_model = surrogate_model.condition_on(eval_x, eval_y)

        tf.random.set_seed(seed + eval_index + 1)
        train_surrogate_model(surrogate_model, model_config, config["run"]["model"])

        np.save(grid_save_path, grid.points.numpy())
        np.save(xs_save_path, surrogate_model.xs.numpy())
        np.save(ys_save_path, surrogate_model.ys.numpy())

        # Remove the simulation files AND the grid from the previous folder,
        # after all important information has been extracted.
        # This is very important, because the simulation directory is effectively useless after,
        # can be regenerated easily and takes up a lot of space.

        # Simulation directory
        shutil.rmtree(os.path.join(eval_dir, task))

        # Previous grid
        prev_grid_save_path = os.path.join(experiment_dir, f"evaluation_{eval_index - 1}", "grid.npy")

        if os.path.exists(prev_grid_save_path):
            os.remove(prev_grid_save_path)

    print("Experiment finised!")


if __name__ == "__main__":
    run_experiment()
