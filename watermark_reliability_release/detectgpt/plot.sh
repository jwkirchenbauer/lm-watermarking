#!/bin/bash

#SBATCH --partition tron

#SBATCH --gres=gpu:rtxa5000:1

#SBATCH --ntasks=4

#SBATCH --mem=32G

#SBATCH --account=nexus

#SBATCH --qos=default

#SBATCH --time=48:00:00

#SBATCH --array=0

#SBATCH --output=slurm_logs/%A_%a.out

#SBATCH --job-name=gen-small

source ~/.bashrc
conda activate watermarking-dev

OUTPUT_DIR=/cmlscratch/manlis/test/watermarking-root/input/new_runs

model_name="facebook/opt-1.3b"
data_name="test_len_200_opt1_3b"
token_len=200
chunk_size=32
data_path="/cmlscratch/manlis/test/watermarking-root/input/new_runs/test_len_200_opt1_3b_evaluation/gen_table_w_metrics.jsonl"
split="no_wm"


python make_plot.py \
    --n_perturbation_list="1,10,100" \
    --base_model_name=${model_name} \
    --data_name=${data_name} \
    --data_path=${data_path} \
    --token_len=${token_len} \
    --chunk_size=${chunk_size} \
    --data_split=${split};

    

