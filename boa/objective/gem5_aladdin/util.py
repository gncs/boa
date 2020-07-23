import os
import subprocess
import json
from typing import Dict, Tuple, List, NamedTuple

from string import Template

import tensorflow as tf

from .process import parse_file, output_regexps


class Gem5DockerRunConfig(NamedTuple):
    container_name: str
    simulation_dir: str


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


def get_output_vector_from_dict(output_dict, output_order):
    output_list = [output_dict[output_name] for output_name in output_order]
    output_vec = tf.convert_to_tensor(output_list)
    output_vec = tf.cast(output_vec, tf.float64)

    return output_vec


def run_gem5_simulation_with_configuration(simulation_dir: str,
                                           run_sh_subdir: str,
                                           simulation_config: str,
                                           output_order: List[str],
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
                generate_design_sweeps_py_path,
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

    result_vec = get_output_vector_from_dict(results, output_order=output_order)

    return result_vec
