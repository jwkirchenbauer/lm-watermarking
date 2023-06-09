# Script to run the generation, attack, and evaluation steps of the pipeline

# requires some OUTPUT_DIR to be set in the environment
# as well as a path to the hf format LLAMA model

RUN_NAME=llama_N500_T200

GENERATION_OUTPUT_DIR="$OUTPUT_DIR"/"$RUN_NAME"

echo "Running generation pipeline with output dir: $GENERATION_OUTPUT_DIR"

python generation_pipeline.py \
    --model_name=$LLAMA_PATH \
    --dataset_name=c4 \
    --dataset_config_name=realnewslike \
    --max_new_tokens=200 \
    --min_prompt_tokens=50 \
    --min_generations=500 \
    --input_truncation_strategy=completion_length \
    --input_filtering_strategy=prompt_and_completion_length \
    --output_filtering_strategy=max_new_tokens \
    --seeding_scheme=selfhash \
    --gamma=0.25 \
    --delta=2.0 \
    --run_name="$RUN_NAME"_gen \
    --wandb=True \
    --verbose=True \
    --output_dir=$GENERATION_OUTPUT_DIR

python attack_pipeline.py \
    --attack_method=gpt \
    --run_name="$RUN_NAME"_gpt_attack \
    --wandb=True \
    --input_dir=$GENERATION_OUTPUT_DIR \
    --verbose=True

python evaluation_pipeline.py \
    --evaluation_metrics=all \
    --run_name="$RUN_NAME"_eval \
    --wandb=True \
    --input_dir=$GENERATION_OUTPUT_DIR \
    --output_dir="$GENERATION_OUTPUT_DIR"_eval \
    --roc_test_stat=all