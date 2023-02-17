from submitit import AutoExecutor
from submitit.helpers import CommandFunction
from itertools import chain
import os
from submitit_utils import ParameterGrid
import argparse

# a debug/dry-run command
dummy_func = CommandFunction(["echo"], verbose=True)

###############################################################################
# Experiment specific command and parameter setup
# (the structure is general, but the values are not)
###############################################################################

base_run_name = None

ROOT_DIR = f'{os.getenv("ROOT_DIR")}'
# OUTPUT_DIR = f'{os.getenv("OUTPUT_DIR")}'
# OUTPUT_DIR = f'{os.getenv("OUTPUT_DIR")}_large_sweep'
# OUTPUT_DIR = f'{os.getenv("OUTPUT_DIR")}_large_sweep_downsize'
# OUTPUT_DIR = f'{os.getenv("OUTPUT_DIR")}_greedy_redo'
OUTPUT_DIR = f'{os.getenv("OUTPUT_DIR")}_greedy_more_gammas'

# starting command/program to which we will append arguments
cmdline_function = CommandFunction(["python"], verbose=True)

# script name
script_name = "run_watermarking.py"

# base args
base_script_args = {
    # "model_name"         :"facebook/opt-2.7b",
    "model_name"         :"facebook/opt-1.3b",
    "dataset_name"       :"c4",
    "dataset_config_name":"realnewslike",
    # "dataset_config_name":"en",
    # "dataset_name": "cml_pile",
    # "dataset_config_name": "all_train_00",
    # "shuffle_dataset"    :"True", # NOTE
    "dynamic_seed"       :"markov_1",
    "store_spike_ents"   :"True",
    # "oracle_model_name"  :"EleutherAI/gpt-j-6B",
    "oracle_model_name"  :"facebook/opt-2.7b",
    "no_wandb"           :"False",
}

# dynamic/hparam args
# i.e. the parameters we would like to cross and sweep over
hparam_sets = [
    # # main sampling sweep, central data
    # {
    #     "min_prompt_tokens": [50],
    #     "max_new_tokens": [200],
    #     "input_truncation_strategy": ["completion_length"],
    #     "input_filtering_strategy": ["prompt_and_completion_length"],
    #     "output_filtering_strategy": ["max_new_tokens"],
    #     "limit_indices": [500],
    #     "bl_logit_bias": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0],
    #     "bl_proportion": [0.1, 0.25, 0.5, 0.75, 0.9],
    #     "bl_type": ["soft"],
    #     "num_beams": [1],
    #     "use_sampling": [True],
    #     "sampling_temp": [0.7],
    # },
    # greedy and beams secondary demos
    # {
    #     "min_sample_tokens":[0],
    #     "min_prompt_tokens": [200],
    #     "max_new_tokens": [500],
    #     "all_gas_no_eos": [True],
    #     "input_truncation_strategy": ["prompt_length"],
    #     "input_filtering_strategy": ["prompt_and_completion_length"],
    #     "output_filtering_strategy": ["no_filter"],
    #     "limit_indices": [500],
    #     "bl_logit_bias": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    #     "bl_proportion": [0.5],
    #     "bl_type": ["soft"],
    #     "num_beams": [1],
    #     "use_sampling": [False],
    #     "sampling_temp": [0.0],
    # },
    # {
    #     "min_sample_tokens":[0],
    #     "min_prompt_tokens": [200],
    #     "max_new_tokens": [500],
    #     "all_gas_no_eos": [True],
    #     "no_repeat_ngram_size": [0],
    #     "input_truncation_strategy": ["prompt_length"],
    #     "input_filtering_strategy": ["prompt_and_completion_length"],
    #     "output_filtering_strategy": ["no_filter"],
    #     "limit_indices": [500],
    #     "bl_logit_bias": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    #     "bl_proportion": [0.5],
    #     "bl_type": ["soft"],
    #     "num_beams": [4],
    #     "use_sampling": [False],
    #     "sampling_temp": [0.0],
    # },
    {
        "min_sample_tokens":[0],
        "min_prompt_tokens": [200],
        "max_new_tokens": [500],
        "all_gas_no_eos": [True],
        "no_repeat_ngram_size": [0],
        "input_truncation_strategy": ["prompt_length"],
        "input_filtering_strategy": ["prompt_and_completion_length"],
        "output_filtering_strategy": ["no_filter"],
        "limit_indices": [500],
        "bl_logit_bias": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        # "bl_logit_bias": [2.0, 5.0, 10.0],
        # "bl_proportion": [0.5],
        # "bl_proportion": [0.75],
        "bl_proportion": [0.9],
        "bl_type": ["soft"],
        "num_beams": [8],
        "use_sampling": [False],
        "sampling_temp": [0.0],
    },
    ############
]

# logic to set derived arguments based on existing arguments in the sweep sets
# the unique run name is the canonical example
def add_conditional_params(param_dict):

    # unique_name = f'{base_run_name+"_" if base_run_name else ""}{param_dict.get("model_name")}_{param_dict.get("dataset_name")}_{param_dict.get("dataset_config_name")}'
    unique_name_keys = ["model_name",
                        "bl_type",
                        "dynamic_seed",
                        "bl_proportion",
                        "bl_logit_bias",
                        "num_beams",
                        "use_sampling",
                        "sampling_temp",
                        "dataset_name",
                        "dataset_config_name",
                        "min_prompt_tokens",
                        "max_new_tokens",
                        "input_truncation_strategy",
                        "input_filtering_strategy",
                        "output_filtering_strategy",
                        "limit_indices",
                        "oracle_model_name"]

    unique_name = f'{base_run_name+"_" if base_run_name else ""}{"_".join([str(param_dict.get(k)) for k in unique_name_keys])}'
    unique_name = unique_name.replace("/", "-").replace(".","-")
    param_dict.update({"run_name": unique_name})
    param_dict.update({"output_dir": f'{OUTPUT_DIR}/{param_dict["run_name"]}'})

# Queue up all the arguments
def add_params(param_dicts):
    new_dicts = []
    for i, param_dict in enumerate(param_dicts):
        new_dict = {}

        new_dict.update({script_name : ""}) # This requires parse block change in submitit.core.utils.py L320
        new_dict.update(base_script_args)
        
        new_dict.update(param_dict)
        add_conditional_params(new_dict)

        new_dicts.append(new_dict)
    return new_dicts

###############################################################################
# Generic submitit and slurm workflow
###############################################################################

# set up the executor and sbatch settings
# executor = AutoExecutor(cluster='slurm', folder=f'{ROOT_DIR}/logs/')
# executor = AutoExecutor(cluster='slurm', folder=f'{ROOT_DIR}/logs_large_sweep/')
# executor = AutoExecutor(cluster='slurm', folder=f'{ROOT_DIR}/logs_large_sweep_downsize/')
# executor = AutoExecutor(cluster='slurm', folder=f'{ROOT_DIR}/logs_greedy_redo/')
executor = AutoExecutor(cluster='slurm', folder=f'{ROOT_DIR}/logs_greedy_more_gammas/')

executor.update_parameters(
    stderr_to_stdout=True,
    slurm_name='water',
    # slurm_account='tomg',
    # slurm_qos='very_high',
    # slurm_qos='high',
    slurm_mem= '52gb', 
    slurm_gres='gpu:rtxa6000:1',
    slurm_time='14:00:00',
    slurm_account='scavenger',
    slurm_partition='scavenger',
    slurm_qos='scavenger',
    # slurm_mem= '32gb', 
    # slurm_cpus_per_task=4,
    # slurm_gres='gpu:rtxa5000:1',
    # slurm_time='12:00:00',
)

# cross and line up parameter combinations
arg_dicts = list(chain(*(ParameterGrid(p_set) for p_set in hparam_sets)))

# set params and apply any extra param logic
arg_dicts = add_params(arg_dicts)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dry_run",
        action="store_true",
        help="just echo the commands to be run",
    )
    args = parser.parse_args()

    # context to make this loop/list comp execute an array job 
    # rather than individual jobs
    with executor.batch():
        
        if args.dry_run:
            fn = dummy_func
        else:
            fn = cmdline_function
        jobs = [executor.submit(fn, **arg_dict) for arg_dict in arg_dicts]

    for job,args in zip(jobs, arg_dicts):
        print(f"Job={job} | uid={args['run_name']}")