#!/bin/bash

#SBATCH --partition tron

#SBATCH --gres=gpu:rtxa6000:1

#SBATCH --ntasks=4

#SBATCH --mem=32G

#SBATCH --account=nexus

#SBATCH --qos=default

#SBATCH --time=48:00:00

#SBATCH --array=0-1

#SBATCH --output=slurm_logs/%A_%a.out

#SBATCH --job-name=run-detect

source ~/.bashrc
conda activate watermarking-dev

OUTPUT_DIR=/cmlscratch/manlis/test/watermarking-root/input/new_runs

# model_name="facebook/opt-1.3b"
# data_path="/cmlscratch/manlis/test/watermarking-root/input/new_runs/test_len_200_opt1_3b_evaluation/gen_table_w_metrics.jsonl"

model_name='facebook/opt-6.7b'
data_path='/cmlscratch/manlis/test/watermarking-root/input/new_runs/test_len_1000_evaluation/gen_table_w_metrics.jsonl'

mask_model="t5-3b"

# token_len=200
chunk_size=32
pct=0.3
split="no_wm"

textlen=600

# python detectgpt_main.py \
#         --n_perturbation_list="10,100" \
#         --do_chunk \
#         --base_model_name=${model_name} \
#         --mask_filling_model_name=${mask_model} \
#         --data_path=/cmlscratch/manlis/test/watermarking-root/input/new_runs/test_len_${textlen}_evaluation/gen_table_w_metrics.jsonl \
#         --token_len=${textlen} \
#         --pct_words_masked=${pct} \
#         --chunk_size=${chunk_size} \
#         --data_split=${split};

declare -a commands

for textlen in 600 1000;
do
    commands+=( "python detectgpt_main.py \
        --n_perturbation_list="10,100" \
        --do_chunk \
        --base_model_name=${model_name} \
        --mask_filling_model_name=${mask_model} \
        --data_path=/cmlscratch/manlis/test/watermarking-root/input/new_runs/test_len_${textlen}_evaluation/gen_table_w_metrics.jsonl \
        --token_len=${textlen} \
        --pct_words_masked=${pct} \
        --chunk_size=${chunk_size} \
        --data_split=${split};" )

done

bash -c "${commands[${SLURM_ARRAY_TASK_ID}]}"

# --data_path=/cmlscratch/manlis/test/watermarking-root/input/new_runs/test_len_${textlen}_evaluation/gen_table_w_metrics.jsonl \


    

