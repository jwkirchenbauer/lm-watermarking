#!/bin/bash

#SBATCH --partition scavenger

#SBATCH --gres=gpu:rtxa6000:1

#SBATCH --ntasks=4

#SBATCH --mem=32G

#SBATCH --account=scavenger

#SBATCH --qos=scavenger

#SBATCH --time=24:00:00

#SBATCH --array=0-2

#SBATCH --output=slurm_logs/no_wm_attack_%A_%a.out

#SBATCH --job-name=run-detect

source ~/.bashrc
conda activate watermarking-dev

# OUTPUT_DIR=/cmlscratch/manlis/test/watermarking-root/input/new_runs

# model_name="facebook/opt-1.3b"
# data_path="/cmlscratch/manlis/test/watermarking-root/input/new_runs/test_len_200_opt1_3b_evaluation/gen_table_w_metrics.jsonl"

# model_name='facebook/opt-6.7b'
# data_path='/cmlscratch/manlis/test/watermarking-root/input/core_simple_1_50_200_gen/gen_table.jsonl'
# data_path=input/core_simple_1_200_1000_gen_prefixes/gen_table_prefixes_200.jsonl
model_name='/cmlscratch/manlis/test/watermarking-root/local_model/llama-7b-base'

mask_model="t5-3b"

# token_len=200
chunk_size=32 # can run 32 when textlen=200
pct=0.3
# split="no_wm"
split='no_wm_paraphrase'

declare -a commands


# for textlen in 50 100 200;
for textlen in 50 100 200;
do
    commands+=( "python detectgpt_main.py \
        --n_perturbation_list='100' \
        --do_chunk \
        --base_model_name=${model_name} \
        --mask_filling_model_name=${mask_model} \
        --filter='null' \
        --data_path=/cmlscratch/manlis/test/watermarking-root/input/core_simple_1_200_1000_no_wm_gpt_p4_prefixes/gen_table_prefixes_${textlen}.jsonl \
        --token_len=${textlen} \
        --pct_words_masked=${pct} \
        --chunk_size=${chunk_size} \
        --data_split=${split};" )
done

bash -c "${commands[${SLURM_ARRAY_TASK_ID}]}"
