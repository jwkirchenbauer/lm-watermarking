# Basic imports
import sys
import os
import argparse
from typing import List, Iterable, Optional
from functools import partial
import time

from tqdm import tqdm
import random
import math
from statistics import mean

import numpy as np
import torch
from torch import Tensor
from tokenizers import Tokenizer

import wandb
import matplotlib.pyplot as plt

# cache path before HF imports just for kicks
# bc I don't really know when this is pulled by the library
# TODO change to passing as an arg to the model load fn
USER = "jkirchen"
# Huggingface cache
HF_HOME=f"/cmlscratch/{USER}/.cache/huggingface"
# HF_HOME=f"/scratch0/{USER}/.cache/huggingface"
# HF_HOME=f"/scratch1/{USER}/.cache/huggingface"
os.environ["HF_HOME"] = HF_HOME

print(os.environ["HF_HOME"])

# HF classses
from transformers import (AutoTokenizer, 
                          AutoModelForSeq2SeqLM, 
                          AutoModelForCausalLM,
                          LogitsProcessorList)

from datasets import load_dataset, Dataset

# watermarking micro lib
from watermark import (BlacklistLogitsProcessor,
                       add_idx,
                       check_input_lengths,
                       check_output_lengths,
                       tokenize_for_generation,
                       generate_completions,
                       evaluate_generation_fluency)

# better bool flag type for argparse
from submitit_utils import str2bool

# some file i/o helpers
from io_utils import write_jsonlines, write_json, read_jsonlines, read_json

def main(args):

    ###########################################################################
    # Start logging
    ###########################################################################
    if not args.no_wandb:

        # storing slurm info to be sent to wandb to allow auditing logfiles later
        args.SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
        args.SLURM_ARRAY_JOB_ID = os.getenv("SLURM_ARRAY_JOB_ID")
        args.SLURM_ARRAY_TASK_ID = os.getenv("SLURM_ARRAY_TASK_ID")

        # start a new wandb run to track this experiment, will send data to it later
        run = wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,

            # track hyperparameters and run metadata
            config=args
        )
    
    print(f"Output dir for this run: {args.output_dir}")
    # notify if exists
    if os.path.exists(args.output_dir):
        print(f"Output dir for this run already exists!")
        print(f"Contents: {sorted(os.listdir(args.output_dir))}")
    else:
        # create the output dir where run artifacts are stored
        os.makedirs(args.output_dir)

    ###########################################################################
    # Instantiate model and tokenizer
    ###########################################################################
    hf_model_name = args.model_name

    if "t5" in hf_model_name or "T0" in hf_model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_model_name)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    # defaults to device 0
    # will need to use 'parallelize' for multi-gpu sharding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    ###########################################################################
    # Load the dataset
    ###########################################################################

    dataset_name, dataset_config_name = args.dataset_name, args.dataset_config_name

    if dataset_name == "cml_pile":
        subsets = [dataset_config_name]
        dataset = load_dataset("input/cml_pile.py",
                                subsets=subsets,
                                streaming=True,
                                split=None,
                                ignore_verifications=True)["train"]
    else:
        dataset = load_dataset(dataset_name, dataset_config_name, split="train", streaming=True)
    
    # log an example
    ds_iterator = iter(dataset)
    idx = 75 # if this is c4, it's the schumacher example lol
    i = 0
    while i < idx: 
        next(ds_iterator)
        i += 1

    example = next(ds_iterator)
    print(example)

    ###########################################################################
    # Construct the blacklist processor/sampler
    ###########################################################################

    all_token_ids = list(tokenizer.get_vocab().values())
    vocab_size = len(all_token_ids)
    print(f"Vocabulary size: {vocab_size}")

    max_new_tokens = args.max_new_tokens
    min_prompt_tokens = args.min_prompt_tokens

    init_seed = args.initial_seed
    dyna_seed=args.dynamic_seed # type not value
    bl_proportion = args.bl_proportion
    bl_logit_bias = args.bl_logit_bias
    bl_type = args.bl_type
    n_beams = args.num_beams
    early_stopping = args.early_stopping
    no_repeat_ngram_size = args.no_repeat_ngram_size
    store_bl_ids = args.store_bl_ids
    store_spike_ents = args.store_spike_ents

    bl_processor = BlacklistLogitsProcessor(bad_words_ids=None, 
                                            store_bl_ids=store_bl_ids, 
                                            store_spike_ents=store_spike_ents, 
                                            eos_token_id=tokenizer.eos_token_id, 
                                            vocab=all_token_ids, 
                                            vocab_size=vocab_size, 
                                            bl_proportion=bl_proportion,
                                            bl_logit_bias=bl_logit_bias,
                                            bl_type=bl_type, 
                                            initial_seed=init_seed, 
                                            dynamic_seed=dyna_seed)                                           

    logit_processor_lst = LogitsProcessorList([bl_processor])

    # Greedy and basic beam search, default
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens, 
        num_beams=n_beams,
    )
    if n_beams > 1:
        # these are only for beam search repetition correction
        if no_repeat_ngram_size > 0:
            gen_kwargs.update(dict(no_repeat_ngram_size=no_repeat_ngram_size))
        gen_kwargs.update(dict(early_stopping=early_stopping))

    if args.use_sampling:
        gen_kwargs.update(dict(do_sample=True,
                                top_k=0,
                                temperature=args.sampling_temp))
    if args.all_gas_no_eos:
        gen_kwargs.update(dict(suppress_tokens=[tokenizer.eos_token_id]))

    generate_without_blacklist = partial(
        model.generate,
        **gen_kwargs
    )
    generate_with_blacklist = partial(
        model.generate,
        logits_processor=logit_processor_lst, 
        **gen_kwargs
    )

    ###########################################################################
    # Construct the generation and measurement pipeline (lazy)
    # that pulls from the streaming dataset, applies the generations map funcs
    ###########################################################################

    # Set up the pipeline functions
    if "c4" in dataset_name:
        columns_to_remove = ["text","timestamp","url"]
    else:
        columns_to_remove = []

    # Construct the data filtering/sampling scheme partials
    token_kwargs = dict(
        hf_model_name=hf_model_name,
        tokenizer=tokenizer,
        model=model,
    )
    if args.input_truncation_strategy == "prompt_length":
        token_kwargs.update(dict(min_prompt_tokens=min_prompt_tokens))
    elif args.input_truncation_strategy == "completion_length":
        token_kwargs.update(dict(max_new_tokens=max_new_tokens))
    else:
        ValueError(f"Unknown input truncation strategy {args.input_truncation_strategy}")
    tokenize_prompts = partial(
        tokenize_for_generation,
        **token_kwargs
    )

    input_check_kwargs = dict(
        # min_sample_len = min_prompt_tokens + max_new_tokens,
        min_sample_len = args.min_sample_tokens, # first line is a bug sometimes with large amounts
    )
    if args.input_filtering_strategy == "prompt_length":
        input_check_kwargs.update(dict(min_prompt_len = min_prompt_tokens,
                                       min_completion_len = 0))
    elif args.input_filtering_strategy == "completion_length":
        input_check_kwargs.update(dict(min_prompt_len = 0,
                                       min_completion_len = max_new_tokens))
    elif args.input_filtering_strategy == "prompt_and_completion_length":
        input_check_kwargs.update(dict(min_prompt_len = min_prompt_tokens,
                                       min_completion_len = max_new_tokens))
    else:
        ValueError(f"Unknown input filtering strategy {args.input_filtering_strategy}")
    input_check = partial(
        check_input_lengths,
        **input_check_kwargs
    )

    if args.output_filtering_strategy == "max_new_tokens":
        output_kwargs = dict(min_output_len = max_new_tokens)
    elif args.output_filtering_strategy == "no_filter":
        output_kwargs = dict(min_output_len = 0)
    else:
        ValueError(f"Unknown output filtering strategy {args.output_filtering_strategy}")
    output_check = partial(
        check_output_lengths,
        **output_kwargs
    )

    gen_completions = partial(
        generate_completions,
        max_new_tokens=max_new_tokens,
        hf_model_name=hf_model_name,
        tokenizer=tokenizer,
        model=model,
        no_bl_partial=generate_without_blacklist,
        w_bl_partial=generate_with_blacklist,
        bl_processor_list=logit_processor_lst,
    )

    ###########################################################################
    # Compose/apply the pipeline steps
    ###########################################################################

    # Apply the pipeline operations to the dataset
    indexed_dataset = dataset.map(add_idx, batched=False, with_indices=True)
    
    # shuffled the first shuffle_buffer_size rows of the (streaming) dataset
    if args.shuffle_dataset:
        shuffled_dataset = indexed_dataset.shuffle(seed=args.shuffle_seed, 
                                                   buffer_size=args.shuffle_buffer_size)
    else:
        shuffled_dataset = indexed_dataset

    # tokenize and truncate the row inputs to create prompts according to the strategy spec'd above
    tokenized_and_truncated_dataset = shuffled_dataset.map(tokenize_prompts, 
                                                           batched=False, 
                                                           with_indices=True)

    # filter the rows of the dataset based on length checks for the tokenized prompts and baseline completions
    input_length_filtered_dataset = tokenized_and_truncated_dataset.filter(input_check, 
                                                                           batched=False, 
                                                                           with_indices=True)

    # perform generation by calling the models
    columns_to_remove += ["inputs", "untruncated_inputs"] # these are now materialized and must be dropped externally
    generations_dataset = input_length_filtered_dataset.map(gen_completions, 
                                                            batched=False, 
                                                            with_indices=True, 
                                                            remove_columns=columns_to_remove)

    # # filter the dataset a last time based on the lengths of the outputs of the model
    # output_length_filtered_dataset = generations_dataset.filter(output_check, 
    #                                                             batched=False, 
    #                                                             with_indices=True)

    ###########################################################################
    # Main loop - actually executes the generation pipeline.
    # and accumulates the result rows in a list, assumes list is "small"-ish
    # and we aren't accumulating any tensors or other memory hogging artifacts
    ###########################################################################
    if not args.load_prev_generations:

        processed_examples = []
        ds_iterator = iter(generations_dataset)
        i = 0
        while i < args.limit_indices:

            ex = next(ds_iterator)
            
            # log basics to stdout
            print(f"#"*80)
            print(f"dataset index: {ex['idx']}")
            print(f"orig_sample_length: {ex['orig_sample_length']}")
            print(f"prompt_length: {ex['prompt_length']}")
            print(f"real_completion_length: {ex['real_completion_length']}")
            print(f"no_bl_num_tokens_generated: {ex['no_bl_num_tokens_generated']}")
            print(f"w_bl_num_tokens_generated: {ex['w_bl_num_tokens_generated']}")

            print(f"\ntruncated_input: ")
            print(ex["truncated_input"])
            print(f"\nbaseline_completion: ")
            print(ex["baseline_completion"])
            print(f"\nno_bl_output: ")
            print(ex["no_bl_output"])
            print(f"\nw_bl_output: ")
            print(ex["w_bl_output"])
            print(f"\nno_bl_gen_time: ")
            print(ex["no_bl_gen_time"])
            print(f"\nno_bl_sec_per_tok: ")
            print(ex["no_bl_sec_per_tok"])
            print(f"\nno_bl_tok_per_sec: ")
            print(ex["no_bl_tok_per_sec"])
            print(f"\nw_bl_gen_time: ")
            print(ex["w_bl_gen_time"])
            print(f"\nw_bl_sec_per_tok: ")
            print(ex["w_bl_sec_per_tok"])
            print(f"\nw_bl_tok_per_sec: ")
            print(ex["w_bl_tok_per_sec"])

            processed_examples.append(ex)
            if output_check(ex) == True:
                i += 1
            else:
                print(f"\nGeneration too short, saving outputs, but not incrementing counter...\n",
                      f"{i} of {len(processed_examples)} rows were satisfactory so far",
                      f"current generation overhead ratio: {round(len(processed_examples)/(i+1), 3)}",
                      f"completed {round(i/args.limit_indices, 2)} of total")
    
    print(f"#"*80,
          f"\nGeneration output length check overhead was num rows processed={len(processed_examples)}",
          f"for {args.limit_indices} samples. Ratio: {round(len(processed_examples)/args.limit_indices, 3)}")
    
    ###########################################################################
    # Generation jsonl dumping/loading
    ###########################################################################

    gen_table_meta_path = f"{args.output_dir}/gen_table_meta.json"
    gen_table_path = f"{args.output_dir}/gen_table.jsonl"
    safe_gen_table_path = f"{args.output_dir}/gen_table_safe.jsonl"

    args.gen_table_already_existed = False

    if not args.load_prev_generations:
        
        if os.path.exists(gen_table_path): 
            print(f"Found existing generation files at this output dir: {args.output_dir}")
            print(f"Writing generations at alternate, safe path and exiting. Note! this only works once. "
                  f"Safe version will get overwritten next time ... ")
            gen_table_path = f"{args.output_dir}/gen_table_safe.jsonl"
            args.gen_table_already_existed = True

        gen_table_meta = args.__dict__
        gen_table = processed_examples
                
        write_jsonlines(gen_table, gen_table_path)
        write_json(gen_table_meta,gen_table_meta_path,indent=4)

        if args.gen_table_already_existed: 
            # finish the wandb run
            if not args.no_wandb: run.finish()
            return # from main, for safety
    else:
        print(f"Loading previously generated outputs for evaluation via oracle model and metrics...")

        assert os.path.exists(gen_table_meta_path), f"failed file check for prev generations metadata json file: {gen_table_meta_path}"
        assert os.path.exists(gen_table_path), f"failed file check for prev generations jsonl file: {gen_table_path}"

        curr_gen_table_meta = args.__dict__.copy()
        prev_gen_table_meta = read_json(gen_table_meta_path)
        
        assert not prev_gen_table_meta["gen_table_already_existed"], f"failed for safety bc 'gen_table_already_existed' was true in the metadata file in this dir, indicating a possible issue"
        assert not os.path.exists(safe_gen_table_path), f"failed for safety bc there is a secondary 'safe' marked file in this dir indicating a possible issue"
        
        params_to_ignore = ["load_prev_generations","SLURM_JOB_ID","SLURM_ARRAY_JOB_ID","SLURM_ARRAY_TASK_ID"]
        for k in params_to_ignore:
            del curr_gen_table_meta[k]
            del prev_gen_table_meta[k]
        assert curr_gen_table_meta == prev_gen_table_meta, "failed safety check that current script params equal the params for the prev generations being loaded"

        # gen_table_meta = argparse.Namespace(**args.__dict__)
        gen_table_meta = args
        gen_table = [ex for ex in read_jsonlines(gen_table_path)]

    if args.generate_only: 
        # finish the wandb run
        if not args.no_wandb: run.finish()
        return # early exit, will reload later for ppl scoring

    # Create a new dataset object either from the loop over examples
    # or from the reloaded json lines
    
    # gen_table_ds = Dataset.from_generator(ex for ex in gen_table) # hack since from_list is newer, and had 2.4.0
    gen_table_ds = Dataset.from_list(gen_table)

    ###########################################################################
    # Perplexity (PPL) evaluation
    # which is a separate step partially bc it requires a different model on gpu
    ###########################################################################
    
    # Load the oracle model for PPL measurement
    # Assume on single GPU and need to free orig model memory for oracle model
    if model is not None:
        model = model.to(torch.device("cpu"))
        del model

    oracle_model_name = args.oracle_model_name
    print(f"Loading oracle model: {oracle_model_name}")
    
    oracle_tokenizer = AutoTokenizer.from_pretrained(oracle_model_name)
    oracle_model = AutoModelForCausalLM.from_pretrained(oracle_model_name).to(device)
    oracle_model.eval()

    # construct fluency/ppl partial
    eval_gen_metrics = partial(
        evaluate_generation_fluency,
        oracle_model_name=oracle_model_name,
        oracle_model=oracle_model,
        oracle_tokenizer=oracle_tokenizer
    )

    print(f"Computing metrics on model generations: {gen_table_ds}")

    gen_table_w_metrics_ds = gen_table_ds.map(eval_gen_metrics, batched=False, with_indices=True)


    print(f"#"*80)
    print(f"baseline avg PPL: {mean(gen_table_w_metrics_ds['baseline_ppl'])}")
    print(f"baseline avg loss: {mean(gen_table_w_metrics_ds['baseline_loss'])}")
    print(f"no_bl avg PPL: {mean(gen_table_w_metrics_ds['no_bl_ppl'])}")
    print(f"no_bl avg loss: {mean(gen_table_w_metrics_ds['no_bl_loss'])}")
    print(f"w_bl avg PPL: {mean(gen_table_w_metrics_ds['w_bl_ppl'])}")
    print(f"w_bl avg loss: {mean(gen_table_w_metrics_ds['w_bl_loss'])}")
    
    # clear the model just for fun
    oracle_model = oracle_model.to(torch.device("cpu"))
    del oracle_model

    gen_table_w_metrics_path = f"{args.output_dir}/gen_table_w_metrics.jsonl"
    if os.path.exists(gen_table_w_metrics_path): 
        print(f"Found existing generation files with metrics added at this output dir. Overwriting anyway :\ -> {args.output_dir}")

    gen_table_w_metrics_lst = [ex for ex in gen_table_w_metrics_ds]
    write_jsonlines(gen_table_w_metrics_lst, gen_table_w_metrics_path)

    # finish the wandb run
    run.finish()

    return 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run watermarked huggingface LM generation pipeline")
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/opt-2.7b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
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
        "--max_new_tokens",
        type=int,
        default=100,
        help="The number of tokens to generate using the model, and the num tokens removed from real text sample",
    )
    parser.add_argument(
        "--min_prompt_tokens",
        type=int,
        default=50, # 500
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
        default=5, # 500
        help="The number of examples (first N) to process from the dataset.",
    )
    parser.add_argument(
        "--input_truncation_strategy",
        type=str,
        default="completion_length",
        choices=["completion_length", "prompt_length"],
        help="The strategy to use when tokenizing and truncating raw inputs to make prompts.",
    )
    parser.add_argument(
        "--input_filtering_strategy",
        type=str,
        default="completion_length",
        choices=["completion_length", "prompt_length", "prompt_and_completion_length"],
        help="The strategy to use when tokenizing and truncating raw inputs to make prompts.",
    )
    parser.add_argument(
        "--output_filtering_strategy",
        type=str,
        default="no_filter",
        choices=["no_filter", "max_new_tokens"],
        help=(f"The strategy to use when filtering/skipping rows if the model didn't ",
              f"generate enough tokens to facilitate analysis.")
    )
    parser.add_argument(
        "--initial_seed",
        type=int,
        default=1234,
        help=("The initial seed to use in the blacklist randomization process.", 
              "Is unused if the process is markov generally. Can be None."),
    )
    parser.add_argument(
        "--dynamic_seed",
        type=str,
        default="markov_1",
        choices=[None, "initial", "markov_1"],
        help="The seeding procedure to use when sampling the blacklist at each step.",
    )
    parser.add_argument(
        "--bl_proportion",
        type=float,
        default=0.5,
        help="The ratio of blacklist to whitelist tokens when splitting the vocabulary",
    )
    parser.add_argument(
        "--bl_logit_bias",
        type=float,
        default=1.0,
        help="The amount of bias (absolute) to add to the logits in the whitelist half of the vocabulary at every step",
    )
    parser.add_argument(
        "--bl_type",
        type=str,
        default="soft",
        choices=["soft", "hard"],
        help="The type of blacklisting being performed.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="The number of beams to use where '1' is no beam search.",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=0,
        # default=8,
        help="ngram size to force the model not to generate, can't be too small or model is handicapped, too large and blows up in complexity.",
    )
    parser.add_argument(
        "--early_stopping",
        type=str2bool,
        default=False,
        help="Whether to use early stopping, only for beam search.",
    )
    # parser.add_argument(
    #     "--hard_min_length",
    #     type=str2bool,
    #     default=False,
    #     help="Whether to use the min length logits processor to force the generations to be max_new_tokens.",
    # )
    parser.add_argument(
        "--oracle_model_name",
        type=str,
        default="EleutherAI/gpt-j-6B",
        help="PPL scoring, or oracle model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--no_wandb",
        type=str2bool,
        default=False,
        help="Whether to log to wandb.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="lm-blacklisting",
        help="The name of the wandb project.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="jwkirchenbauer",
        help="The wandb entity/user for the project.",
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
        "--load_prev_generations",
        type=str2bool,
        default=False,
        help=("Whether to run generations or load from a json lines in the output_dir. "
             "If True, this file must exist and meta/args must match"),
    )
    parser.add_argument(
        "--store_bl_ids",
        type=str2bool,
        default=False,
        help=("Whether to store all the blacklists while generating with bl processor. "),
    )
    parser.add_argument(
        "--store_spike_ents",
        type=str2bool,
        default=False,
        help=("Whether to store the spike entropies while generating with bl processor. "),
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
        "--generate_only",
        type=str2bool,
        default=False,
        help=("Whether to only produce outputs and not evaluate anything like ppl"),
    )
    parser.add_argument(
        "--all_gas_no_eos",
        type=str2bool,
        default=False,
        help=("Whether to weight the EOS token as -inf"),
    )
    
    args = parser.parse_args()

    main(args)

