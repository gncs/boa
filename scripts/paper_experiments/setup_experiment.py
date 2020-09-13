"""
This script will create the required structure so that exhaustive tests can be ran for a task.
"""

import os
import shutil
import subprocess
from sacred import Experiment

from string import Template

ex = Experiment("setup_boa_experiments")

machsuite_tasks = [
    "fft_transpose",
    "stencil_stencil3d",
    "gemm_blocked"
]

smaug_tasks = [
    "minerva"
]

methods = [
    "sgs",
    "ff-gp",
    "gpar-targets",
    "gpar-full",
    "bnn"
]


@ex.config
def config():

    # Name of the task for which we will generate
    task = ""

    # Path to where the experiment folders should be generated
    experiment_root_path = "/rds/user/gf332/hpc-work/BayesOpt/experiments"

    # Path to where the template files are located
    resources_path = "/rds/user/gf332/hpc-work/boa/scripts/paper_experiments/resources"

    # Number of experiments per task
    num_experiments = 50

    # Number of warmup points to use per task
    num_warmup_points = 10

    # Total number of evaluation to take
    max_num_evaluations = 70


@ex.automain
def setup_experiment(task,
                     experiment_root_path,
                     resources_path,
                     num_experiments,
                     num_warmup_points,
                     max_num_evaluations,
                     _log,
                     warmup_folder_name="warmups",
                     ):

    # =========================================================================
    # Initial checks
    # =========================================================================
    if task not in (machsuite_tasks + smaug_tasks):
        raise ValueError(f"Task must be one of {machsuite_tasks + smaug_tasks}, but '{task}' was given!")

    experiment_base_path = os.path.join(experiment_root_path, task)

    if os.path.exists(experiment_base_path):
        _log.info(f"Experiment folder for task: {task} already exists at {experiment_base_path}, skipping!")

        return

    # =========================================================================
    # Build folder structure
    # =========================================================================

    # Create base folder for the task
    os.makedirs(experiment_base_path, exist_ok=False)

    # Create base folder for the warmups if it doesn't exist yet
    # Note: this folder will be shared across all methods for the same task
    warmup_folder_base_path = os.path.join(experiment_base_path, warmup_folder_name)

    if not os.path.exists(warmup_folder_base_path):
        _log.info(f"Warmup folders for task {task} do not exist yet, creating them now!")
        os.makedirs(warmup_folder_base_path, exist_ok=False)

        for index in range(num_experiments):
            warmup_folder_path = os.path.join(warmup_folder_base_path, f"warmup_{index}")
            os.makedirs(warmup_folder_path, exist_ok=False)

    for method in methods:
        method_path = os.path.join(experiment_base_path, method)
        os.makedirs(method_path, exist_ok=False)

        # Write SLURM run script
        with open(os.path.join(resources_path, "slurm_array.template")) as slurm_template_file:
            slurm_content_template = Template(slurm_template_file.read())

        slurm_content = slurm_content_template.substitute({"task": task,
                                                           "method": method})

        with open(os.path.join(experiment_base_path, f"run_{method}"), "w") as slurm_script:
            slurm_script.write(slurm_content)

        # Create the individual experiment folders
        for index in range(num_experiments):

            experiment_path = os.path.join(method_path, f"experiment_{index}")
            os.makedirs(experiment_path)

            # ---------------------------------------------------------------------
            # MachSuite Tasks
            # ---------------------------------------------------------------------
            if task in machsuite_tasks:

                # Copy designsweeptypes.py
                sweeptypes_name = "designsweeptypes.py"
                template_file_name = "machsuite_template.xe"

                shutil.copy(src=os.path.join(resources_path, sweeptypes_name),
                            dst=os.path.join(experiment_path, sweeptypes_name))

                # Copy the design template
                shutil.copy(src=os.path.join(resources_path, template_file_name),
                            dst=os.path.join(experiment_path, template_file_name))

                # Copy appropriate MachSuite constants
                with open(os.path.join(experiment_path, "machsuite_constants.xe"), "w") as constants_file:
                    result = subprocess.run(
                        ["grep",
                         task,
                         os.path.join(resources_path, "machsuite_constants.xe")],
                        stdout=constants_file,
                        #text=True,
                        check=True,
                    )

                # Create configuration file
                config_file_path = os.path.join(resources_path, "machsuite_config.pyconf")

                with open(config_file_path) as config_template_file:

                    config_template = Template(config_template_file.read())

                    config = config_template.substitute({"task": task,
                                                         "experiment_dir": experiment_path,
                                                         "warmup_base_dir": os.path.join(warmup_folder_base_path, f"warmup_{index}"),
                                                         "model": method,
                                                         "seed": index,
                                                         "num_warmup_points": num_warmup_points,
                                                         "max_num_evaluations": max_num_evaluations})

                config_target_path = os.path.join(experiment_path, "config.py")
                with open(config_target_path, "w") as config_target:
                    config_target.write(config)

            # ---------------------------------------------------------------------
            # SMAUG Tasks
            # ---------------------------------------------------------------------
            elif task in smaug_tasks:
                raise NotImplementedError

            else:
                raise NotImplementedError