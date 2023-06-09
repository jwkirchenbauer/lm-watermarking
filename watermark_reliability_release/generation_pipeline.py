# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
from functools import partial
from tqdm import tqdm
import wandb

print(f"Current huggingface cache dir: {os.environ['HF_HOME']}")

# HF classses
from transformers import LogitsProcessorList, DataCollatorWithPadding

# better bool flag type for argparse
from utils.submitit import str2bool

# some file i/o helpers
from utils.io import write_jsonlines, write_json

# watermarking functionality
from watermark_processor import WatermarkLogitsProcessor

# generation pipeline helpers
from utils.generation import (
    MAX_GENERATIONS,
    load_model,
    load_hf_dataset,
    check_input_lengths,
    check_output_lengths,
    tokenize_for_generation,
    generate,
)


def main(args):
    ###########################################################################
    # Start logging
    ###########################################################################
    # storing slurm info to allow auditing logfiles later
    args.SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
    args.SLURM_ARRAY_JOB_ID = os.getenv("SLURM_ARRAY_JOB_ID")
    args.SLURM_ARRAY_TASK_ID = os.getenv("SLURM_ARRAY_TASK_ID")

    if args.wandb:
        # start a new wandb run to track this experiment, will send data to it later
        run = wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.run_name}",
            # track hyperparameters and run metadata
            config=args,
            tags=args.wandb_tags,
        )

    ###########################################################################
    # Create the output dir
    ###########################################################################
    print(f"Output dir for this run: {args.output_dir}")
    # notify if exists
    if os.path.exists(args.output_dir):
        print(f"Output dir for this run already exists!")
        print(f"Contents: {sorted(os.listdir(args.output_dir))}")
    else:
        # create the output dir where run artifacts are stored
        os.makedirs(args.output_dir)

    ###########################################################################
    # Load the dataset
    ###########################################################################
    # basic ops like shuffling and select are done in load fn
    dataset = load_hf_dataset(args)

    ###########################################################################
    # Instantiate model and tokenizer
    ###########################################################################

    model, tokenizer, device = load_model(args)

    ###########################################################################
    # Configure the prompt construction partial
    ###########################################################################

    # Construct the data filtering/sampling scheme partials
    token_kwargs = dict(
        hf_model_name=args.model_name_or_path,
        tokenizer=tokenizer,
        args=args,
    )
    if args.input_truncation_strategy == "prompt_length":
        token_kwargs.update(dict(min_prompt_tokens=args.min_prompt_tokens))
    elif args.input_truncation_strategy == "completion_length":
        token_kwargs.update(dict(max_new_tokens=args.max_new_tokens))
    elif args.input_truncation_strategy == "no_truncation":
        # truncate_input_for_prompt is a bool flag, that is set by
        # the dataset loading function, semi-redundant, to make sure
        # people are very aware of which input data style they are using
        assert (
            args.truncate_input_for_prompt == False
        ), "Cannot truncate input for prompt if 'no_truncation' strategy is specified"
        pass
    else:
        ValueError(f"Unknown input truncation strategy {args.input_truncation_strategy}")
    tokenize_prompts = partial(tokenize_for_generation, **token_kwargs)

    ###########################################################################
    # Configure the I/O data validation partials
    ###########################################################################

    input_check_kwargs = dict(
        min_sample_len=args.min_sample_tokens,
        max_input_len=model.config.max_position_embeddings,
        max_new_tokens=args.max_new_tokens,
    )
    if args.input_filtering_strategy == "prompt_length":
        input_check_kwargs.update(dict(min_prompt_len=args.min_prompt_tokens, min_completion_len=0))
    elif args.input_filtering_strategy == "completion_length":
        input_check_kwargs.update(dict(min_prompt_len=0, min_completion_len=args.max_new_tokens))
    elif args.input_filtering_strategy == "prompt_and_completion_length":
        input_check_kwargs.update(
            dict(min_prompt_len=args.min_prompt_tokens, min_completion_len=args.max_new_tokens)
        )
    elif args.input_filtering_strategy == "no_filter":
        input_check_kwargs.update(dict(min_prompt_len=0, min_completion_len=0))
    else:
        ValueError(f"Unknown input filtering strategy {args.input_filtering_strategy}")
    input_check = partial(check_input_lengths, **input_check_kwargs)

    if args.output_filtering_strategy == "max_new_tokens":
        output_kwargs = dict(min_output_len=args.max_new_tokens)
    elif args.output_filtering_strategy == "no_filter":
        output_kwargs = dict(min_output_len=0)
    else:
        ValueError(f"Unknown output filtering strategy {args.output_filtering_strategy}")
    output_check = partial(check_output_lengths, **output_kwargs)

    ###########################################################################
    # Construct the watermark processor
    ###########################################################################

    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        delta=args.delta,
        seeding_scheme=args.seeding_scheme,
        store_spike_ents=args.store_spike_ents,
        select_green_tokens=True,
    )

    ###########################################################################
    # Configure the generation partials
    ###########################################################################

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    # FIXME can add typica
    if args.use_sampling:
        gen_kwargs.update(
            dict(
                do_sample=True,
                top_k=args.top_k,
                top_p=args.top_p,
                typical_p=args.typical_p,
                temperature=args.sampling_temp,
            )
        )
    else:
        gen_kwargs.update(dict(num_beams=args.num_beams))

    generate_without_watermark = partial(model.generate, **gen_kwargs)
    generate_with_watermark = partial(
        model.generate, logits_processor=LogitsProcessorList([watermark_processor]), **gen_kwargs
    )

    # construct the collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8)

    generation_partial = partial(
        generate,
        data_collator=data_collator,
        generate_without_watermark=generate_without_watermark,
        generate_with_watermark=generate_with_watermark,
        watermark_processor=watermark_processor,
        tokenizer=tokenizer,
        device=device,
        args=args,
    )

    ###########################################################################
    # Compose the partials to create the pipeline
    ###########################################################################

    # tokenize and truncate the row inputs to create prompts according to the strategy spec'd above
    dataset_w_prompts = dataset.map(tokenize_prompts, batched=False)

    # filter the rows of the dataset based on length checks for the tokenized prompts and baseline completions
    dataset_input_len_filtered = dataset_w_prompts.filter(input_check, batched=False)

    # need to remove the input tensor column after this map
    # bc it persists between the prompt creation and generation maps
    columns_to_remove = args.columns_to_remove + ["input_ids"]

    # call the generation partial on each prompt in the dataset
    dataset_w_generations = dataset_input_len_filtered.map(
        generation_partial,
        batched=True,
        batch_size=args.generation_batch_size,
        remove_columns=columns_to_remove,
    )

    ###########################################################################
    # Main loop - actually executes the generation pipeline.
    # and accumulates the result rows in a list, assumes list is "small"-ish
    # and we aren't accumulating any tensors or other memory hogging artifacts
    ###########################################################################

    processed_examples = []
    ds_iterator = iter(dataset_w_generations)
    i = 0
    total_steps = 0
    pbar = tqdm(total=args.min_generations)
    while i < args.min_generations:
        try:
            ex = next(ds_iterator)
            total_steps += 1
        except StopIteration:
            break

        if args.verbose:
            # log basics to stdout
            print(f"#" * 80)
            print(f"dataset index: {ex['idx']}")
            print(f"orig_sample_length: {ex['orig_sample_length']}")
            print(f"prompt_length: {ex['prompt_length']}")
            print(f"real_completion_length: {ex['baseline_completion_length']}")
            print(f"no_wm_output_length: {ex['no_wm_output_length']}")
            print(f"w_wm_output_length: {ex['w_wm_output_length']}")

            print(f"\ntruncated_input: ")
            print(ex["truncated_input"])
            print(f"\nbaseline_completion: ")
            print(ex["baseline_completion"])
            print(f"\nno_wm_output: ")
            print(ex["no_wm_output"])
            print(f"\nw_wm_output: ")
            print(ex["w_wm_output"])

        processed_examples.append(ex)

        if output_check(ex):
            i += 1
            pbar.update(1)
        else:
            print(
                f"\n{i} of {len(processed_examples)} rows were satisfactory so far, {round(i/args.min_generations, 2)} of total.",
                f"\nCurrent generation overhead ratio: {round(len(processed_examples)/(i+1), 3)}.",
            )
        # if using wandb, log progress to wandb
        if args.wandb:
            run.log(
                {
                    "num_satisfactory_samples": i,
                    "progress_ratio": i / args.min_generations,
                    "generation_overhead_ratio": len(processed_examples) / (i + 1),
                    "total_generated_samples": len(processed_examples),
                },
                step=total_steps,
            )
    pbar.close()

    print(
        f"#" * 80,
        f"\nGeneration output length check overhead was num rows processed={len(processed_examples)}",
        f"for {args.min_generations} samples. Ratio: {round(len(processed_examples)/args.min_generations, 3)}",
    )
    if i < args.min_generations:
        print(
            f"#" * 80,
            f"\nWarning, may have run out of data before {args.min_generations} satisfactory samples were generated. ",
            f"\nNote, raw dataset limit was {args.limit_indices} rows.",
            f"\n{len(processed_examples)} prompt passed input checks and yielded generations, and {i} passed output checks,",
            f"\nProgress made: {round(i/args.min_generations, 2)}",
        )

    ###########################################################################
    # Generation jsonl dumping
    ###########################################################################

    gen_table_meta_path = f"{args.output_dir}/gen_table_meta.json"
    gen_table_path = f"{args.output_dir}/gen_table.jsonl"
    safe_gen_table_path = f"{args.output_dir}/gen_table_safe.jsonl"

    args.gen_table_already_existed = False

    if os.path.exists(gen_table_path):
        args.gen_table_already_existed = True
        print(f"Found existing generation files at this output dir: {args.output_dir}")
        if args.overwrite:
            print("Overwriting old generation files.")
            gen_table_path = gen_table_path
        else:
            print(
                f"Writing generations at alternate, safe path and exiting. Note! this only works once. "
                f"Safe version will get overwritten next time ... "
            )
            gen_table_path = safe_gen_table_path

    gen_table_meta = args.__dict__
    gen_table = processed_examples

    write_jsonlines(gen_table, gen_table_path)
    write_json(gen_table_meta, gen_table_meta_path, indent=4)

    # finish the wandb run
    if args.wandb:
        run.finish()
    return  # reload in separate script for metric measurement


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run watermarked huggingface LM generation pipeline"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-1.3b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=True,
        help="Whether to run model in float16 precsion.",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="c4",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="realnewslike",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="The split of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--stream_dataset",
        type=str2bool,
        default=True,
        help="Whether to stream the dataset from the web or download it locally.",
    )
    parser.add_argument(
        "--columns_to_remove",
        type=str,
        default=None,
        help="Comma separated list of columns to remove from the dataset before generation.",
    )
    parser.add_argument(
        "--shuffle_dataset",
        type=str2bool,
        default=False,
        help="Whether to shuffle the dataset before sampling.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=1234,
        help="The seed to use for dataset shuffle op.",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=10_000,
        help="The buffer size to use for dataset shuffle op - takes n rows first, then shuffles those indices",
    )
    parser.add_argument(
        "--prompt_id",
        type=int,
        default=0,
        help="If the dataset supports multiple instruction prompts, denotes which one to use. 0 is default/no prompt.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="The number of tokens to generate using the model, and the num tokens removed from real text sample",
    )
    parser.add_argument(
        "--min_prompt_tokens",
        type=int,
        default=50,  # 500
        help="The number of examples (first N) to process from the dataset.",
    )
    parser.add_argument(
        "--min_sample_tokens",
        type=int,
        default=0,
        help="The the minimum length of raw prompt samples to consider.",
    )
    parser.add_argument(
        "--limit_indices",
        type=int,
        default=None,
        help="The number of examples (first N) to pull from the dataset, if None, pull all, and then set this arg to the number of rows in the dataset.",
    )
    parser.add_argument(
        "--min_generations",
        type=int,
        default=500,
        help="The minimum number of valid generations according to the output check strat to sample.",
    )
    parser.add_argument(
        "--input_truncation_strategy",
        type=str,
        default="completion_length",
        choices=["no_truncation", "completion_length", "prompt_length"],
        help="The strategy to use when tokenizing and truncating raw inputs to make prompts.",
    )
    parser.add_argument(
        "--input_filtering_strategy",
        type=str,
        default="completion_length",
        choices=["no_filter", "completion_length", "prompt_length", "prompt_and_completion_length"],
        help="The strategy to use when tokenizing and truncating raw inputs to make prompts.",
    )
    parser.add_argument(
        "--output_filtering_strategy",
        type=str,
        default="no_filter",
        choices=["no_filter", "max_new_tokens"],
        help=(
            f"The strategy to use when filtering/skipping rows if the model didn't ",
            f"generate enough tokens to facilitate analysis.",
        ),
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=False,
        help=("Whether to perform sampling during generation. (non-greedy decoding)"),
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="The temperature to use when generating using multinom sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="The top k to use when generating using top_k version of multinom sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="The top p to use when generating using top_p version of sampling",
    )
    parser.add_argument(
        "--typical_p",
        type=float,
        default=1.0,
        help="The typical p to use when generating using typical decoding version of multinom sampling",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="The number of beams to use where '1' is no beam search.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=None,
        help="Seed for setting the torch rng prior to generation using any decoding scheme with randomness.",
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=4,
        help="The batch size to use for generation.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="The seeding procedure to use for the watermark.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="The ratio of tokens to put in the greenlist when splitting the vocabulary",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="The amount of bias (absolute) to add to the logits in the whitelist half of the vocabulary at every step",
    )
    parser.add_argument(
        "--store_spike_ents",
        type=str2bool,
        default=True,
        help=("Whether to store the spike entropies while generating with watermark processor. "),
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=False,
        help="Whether to log the generations to stdout.",
    )
    parser.add_argument(
        "--wandb",
        type=str2bool,
        default=False,
        help="Whether to log to wandb.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="lm-watermarking",
        help="The name of the wandb project.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="jwkirchenbauer",
        help="The wandb entity/user for the project.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="The comma separated list of tags to add to the wandb run.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="The unique name for the run.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="The unique name for the run.",
    )
    parser.add_argument(
        "--overwrite",
        type=str2bool,
        default=False,
        help="Allow overwriting of old generation files at the same output location.",
    )
    args = parser.parse_args()

    ###########################################################################
    # Argument validation and conditional setting
    ###########################################################################
    # for removing some columns to save space
    args.columns_to_remove = args.columns_to_remove.split(",") if args.columns_to_remove else []

    # if decoding scheme is not sampling, then set generation seed to None
    # to avoid confusion and calling the torch rng unnecessarily
    args.generation_seed = args.generation_seed if args.use_sampling else None

    # -1 value for min_generations means no specified minimum
    # with the assumption that the
    if args.min_generations <= 0:
        args.min_generations = MAX_GENERATIONS
        print(
            f"Warning: min_generations is -1. A hardcoded value of {MAX_GENERATIONS} will be used to limit the generation loop."
        )

    if args.limit_indices is None:
        print("No limit_indices specified, pulling all examples from the dataset.")
    else:
        print(f"Limiting iteration to {args.limit_indices} examples from the dataset.")

    # split wandb tags
    if args.wandb_tags != "":
        args.wandb_tags = args.wandb_tags.split(",")
    else:
        args.wandb_tags = []

    main(args)
