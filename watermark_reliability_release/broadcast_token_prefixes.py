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


note = "Note: this script should be moved to/run from the same dir as the `utils` subdir lives in to work properly"
print(note)

import os
import argparse
from functools import partial
from tqdm import tqdm

import wandb
import torch
import numpy as np

from datasets import Dataset, concatenate_datasets

from utils.submitit import str2bool  # better bool flag type for argparse
from utils.io import read_jsonlines, read_json, write_json, write_jsonlines
from utils.evaluation import load_tokenizer, NO_CHECK_ARGS
from utils.generation import tokenize_only

print(f"Current huggingface cache dir: {os.environ['HF_HOME']}")


def main(args):
    ###########################################################################
    # Load generations
    ###########################################################################
    print(f"Input dir for this run: {args.input_dir}")

    print(f"Loading previously generated outputs for evaluation via oracle model and metrics...")

    # check for the "attacked version" of the gen table first
    gen_table_meta_path = f"{args.input_dir}/gen_table_attacked_meta.json"
    gen_table_path = f"{args.input_dir}/gen_table_attacked.jsonl"
    safe_gen_table_path = f"{args.input_dir}/gen_table_attacked_safe.jsonl"

    attack_variants_exist = [
        os.path.exists(gen_table_meta_path),
        os.path.exists(gen_table_path),
    ]
    found_attacked_files = all(attack_variants_exist)
    if not found_attacked_files:
        gen_table_meta_path = f"{args.input_dir}/gen_table_meta.json"
        gen_table_path = f"{args.input_dir}/gen_table.jsonl"
        safe_gen_table_path = f"{args.input_dir}/gen_table_safe.jsonl"

        assert os.path.exists(
            gen_table_meta_path
        ), f"failed file check for prev generations metadata json file: {gen_table_meta_path}"
        assert os.path.exists(
            gen_table_path
        ), f"failed file check for prev generations jsonl file: {gen_table_path}"

    assert not os.path.exists(safe_gen_table_path), (
        f"failed for safety bc there is a secondary 'safe' marked file",
        f" in this dir indicating a possible issue with the generation step. ",
    )

    cmdline_args = args.__dict__.copy()
    prev_gen_table_meta = read_json(gen_table_meta_path)

    joined_args = prev_gen_table_meta.copy()
    for k, v in cmdline_args.items():
        if v is not None or (not k in joined_args):
            joined_args.update({k: v})
        else:
            print(
                f"cmdline arg {k} is None, leaving it as the value found in the input metadata (or None): {prev_gen_table_meta.get(k)}"
            )

    # check that the args used to generate the prev generations are the same as
    # the current args, for the intersection of keys

    for key in prev_gen_table_meta.keys():
        if key in NO_CHECK_ARGS:
            continue
        assert joined_args[key] == prev_gen_table_meta[key], (
            f"failed for safety bc after merging the prev metadata with "
            f"the current cmdline args, values for '{key}' are not the same. "
            f"in metadata: {prev_gen_table_meta[key]}, passed: {cmdline_args[key]}. "
            f"Pass the '--overwrite_args' flag to ignore this check."
        )

    args = argparse.Namespace(**joined_args)
    gen_table = [ex for ex in read_jsonlines(gen_table_path)]
    gen_table_ds = Dataset.from_list(gen_table[: args.limit_rows])

    ######## length filtering: only keeps the samples of exact N tokens ########

    df = gen_table_ds.to_pandas()
    original_len = len(df)
    print(f"Origianl #samples: {original_len}")
    if args.filter_length:
        df = df[
            (df["baseline_completion_length"] == args.max_new_tokens)
            & (df["no_wm_output_length"] == args.max_new_tokens)
            & (df["w_wm_output_length"] == args.max_new_tokens)
        ]
        # TODO: filter length for the attacked output
        print(f" after filtering token length: {len(df)}")
    gen_table_ds = Dataset.from_pandas(df)

    ###########################################################################
    # Prefix list logic
    ###########################################################################
    from utils.generation import tokenize_and_truncate

    print(f"Generating prefixes for the gen table...")

    # load the tokenizer
    tokenizer = load_tokenizer(args)

    def generate_prefix(example, prefix_length=None, text_col_names=None, tokenizer=None):
        assert prefix_length is not None, "prefix_length must be specified"
        assert text_col_names is not None and isinstance(
            text_col_names, list
        ), "text_col_names must be a list of column names"

        # make a copy of the example
        example = example.copy()

        tokd_column_data = {}
        for text_col_name in text_col_names:
            try:
                # check that the col exists
                assert text_col_name in example, f"text_col_name '{text_col_name}' not in example"
                # check whether the prefix is OOB for this example
                # NOTE, this logic might not make perfect sense, but it avoids having prefixes that are ragged
                # which is a better quality when measuring @ idx_T

                # tokenize first because we can't rely on the length col existing
                example = tokenize_only(
                    example,
                    input_col_name=text_col_name,
                    hf_model_name=args.model_name_or_path,
                    tokenizer=tokenizer,
                    model_max_length=args.model_max_length,
                )
                raw_inputs = example.pop("input_ids")

                if not (prefix_length <= raw_inputs.shape[1]):
                    if args.verbose:
                        print(
                            f"Skipping prefix generation for col {text_col_name} because prefix_length"
                            f" {prefix_length} is OOB for this example (orig length={raw_inputs.shape[1]})."
                        )
                    continue

                # else slice the inputs to the prefix length
                inputs = raw_inputs[:, : prefix_length + 1]
                prefix_len = inputs.shape[1]

                # decode the prefix
                decoded_prefix = tokenizer.decode(inputs[0], skip_special_tokens=True)
                # store the prefix and it's length
                tokd_column_data.update(
                    {
                        f"{text_col_name}": decoded_prefix,
                        f"{text_col_name}_length": prefix_len,
                    }
                )
            except Exception as e:
                if args.verbose:
                    print(
                        f"Failed to generate prefix of len {prefix_length} for example idx={example['idx']}\n"
                        f"Should either be becuase the col doesnt exist, or the prefix is OOB for this col in this example."
                    )
                print(f"Exception: {e}")
            if text_col_name not in tokd_column_data:
                tokd_column_data.update({f"{text_col_name}": None, f"{text_col_name}_length": None})

        # add the prefix_len to the example
        # then add the prefixes to the example
        example.update({"prefix_length": prefix_length})
        example.update(tokd_column_data)
        return example

    # if max_prefix_length is not specified, use the max length for the gen table
    if args.max_prefix_length is None:
        # args.max_prefix_length = args.model_max_length
        args.max_prefix_length = args.max_new_tokens

    # get the maximum length out of the ["baseline_completion_length", "no_wm_output_length", "w_wm_output_length", "w_wm_output_attacked_length"]
    # found in the gen table
    max_gen_table_output_length = max(
        [
            ex["baseline_completion_length"]
            for ex in gen_table_ds
            if "baseline_completion_length" in ex
        ]
        + [ex["no_wm_output_length"] for ex in gen_table_ds if "no_wm_output_length" in ex]
        + [ex["w_wm_output_length"] for ex in gen_table_ds if "w_wm_output_length" in ex]
        + [
            ex["w_wm_output_attacked_length"]
            for ex in gen_table_ds
            if "w_wm_output_attacked_length" in ex
        ]
    )

    args.max_prefix_length = min(args.max_prefix_length, max_gen_table_output_length)

    # round down to the nearest multiple of prefix_stride
    last_multiple = args.max_prefix_length - (args.max_prefix_length % args.prefix_stride)
    prefix_lengths = list(
        range(args.prefix_stride, last_multiple + args.prefix_stride, args.prefix_stride)
    )
    # if missing the largest prefix length, add it
    if prefix_lengths[-1] != args.max_prefix_length:
        prefix_lengths.append(args.max_prefix_length)

    if args.max_prefix_length > prefix_lengths[-1]:
        print(
            f"WARNING: max_prefix_length {args.max_prefix_length} is larger than the last prefix length {prefix_lengths[-1]} "
            f"as computed by prefix_stride {args.prefix_stride} multiples up to the longest prefix length in the gen table: "
            f"{max_gen_table_output_length}."
        )

    # store the prefix lengths
    args.prefix_lengths = prefix_lengths
    print(prefix_lengths)

    ###########################################################################
    # Create output dir if it doesn't exist, and warn if it contains metric file
    # we do this here because we need the prefix list
    ###########################################################################
    # gen_table_prefixes_path = f"{args.output_dir}/gen_table_prefixes.jsonl"
    # gen_table_prefixes_meta_path = f"{args.output_dir}/gen_table_prefixes_meta.json"
    # making these the same as normal data so they can be used in the same way by eval
    gen_table_prefixes_path = f"{args.output_dir}/gen_table.jsonl"
    gen_table_prefixes_meta_path = f"{args.output_dir}/gen_table_meta.json"

    if found_attacked_files:
        gen_table_prefixes_path = f"{args.output_dir}/gen_table_attacked.jsonl"
        gen_table_prefixes_meta_path = f"{args.output_dir}/gen_table_attacked_meta.json"

    print(f"Output dir for this run: {args.output_dir}")
    # notify if exists
    if os.path.exists(args.output_dir):
        print(f"Output dir for this run already exists!")
        print(f"Contents: {sorted(os.listdir(args.output_dir))}")
        # warn if metrics file exists
        if args.save_per_prefix:
            for prefix_len in prefix_lengths:
                prefix_table_path = (
                    f"{gen_table_prefixes_path.replace('.jsonl','')}_{prefix_len}.jsonl"
                )
                if os.path.exists(prefix_table_path):
                    if not args.overwrite_output_file:
                        print(
                            f"WARNING: Exiting to avoid overwriting prefix output file. "
                            f"Pass the '--overwrite_output_file' flag to ignore this check."
                        )
                        exit()
                    else:
                        print(
                            f"WARNING: Found existing prefix files at this output dir. "
                            f"Overwriting anyway :/"
                        )

        elif os.path.exists(gen_table_prefixes_path):
            if not args.overwrite_output_file:
                print(
                    f"WARNING: Exiting to avoid overwriting prefix output file. "
                    f"Pass the '--overwrite_output_file' flag to ignore this check."
                )
                exit()
            else:
                print(
                    f"WARNING: Found existing prefix files at this output dir. "
                    f"Overwriting anyway :/"
                )
    else:
        # create the output dir where run artifacts are stored
        os.makedirs(args.output_dir)

    ###########################################################################
    # Generate the prefixes
    ###########################################################################

    prefix_tables = []
    gen_table_ds_lst = [ex for ex in gen_table_ds]

    # hacky check to see whether were working with attacked files
    text_col_names = ["baseline_completion", "no_wm_output", "w_wm_output"]
    if "w_wm_output_attacked" in gen_table_ds_lst[0]:
        assert found_attacked_files, (
            f"found 'w_wm_output_attacked' in the gen table, but apparently we didn't 'load attacked files'?."
            f"Odd... please check whats going on in the input_dir."
        )

        text_col_names.append("w_wm_output_attacked")

    for prefix_len in tqdm(prefix_lengths):
        prefixes_partial = partial(
            generate_prefix,
            prefix_length=prefix_len,
            tokenizer=tokenizer,
            text_col_names=text_col_names,
        )
        gen_table_prefixes = [prefixes_partial(ex) for ex in gen_table_ds_lst]

        # add the prefix dataset to the list of prefix tables
        prefix_tables.append(Dataset.from_list(gen_table_prefixes))

    # now concat the tables
    gen_table_prefixes = concatenate_datasets(prefix_tables)

    ###########################################################################
    # Write the metadata and final dataset out to disk in jsonl format
    # (and optionally save the individual prefix shards)
    ###########################################################################

    # write the metadata
    write_json(args.__dict__, gen_table_prefixes_meta_path, indent=4)

    # write the dataset
    if not args.save_per_prefix:
        write_jsonlines(gen_table_prefixes, gen_table_prefixes_path)
    else:
        # save the individual prefix shards
        for prefix_len in prefix_lengths:
            prefix_table = gen_table_prefixes.filter(lambda ex: ex["prefix_length"] == prefix_len)
            prefix_table_path = f"{gen_table_prefixes_path.replace('.jsonl','')}_{prefix_len}.jsonl"
            write_jsonlines(prefix_table, prefix_table_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform jsonl datasets into a broadcasted prefix version."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="use to load the tokenizer",
    )
    parser.add_argument(
        "--prefix_stride",
        type=int,
        default=10,
        help="The stride to use when generating prefixes.",
    )
    parser.add_argument(
        "--max_prefix_length",
        type=int,
        default=None,
        help="The maximum prefix length to use when generating prefixes.",
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="The directory containing the input files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=("The directory in which to write out the dataset after creating prefixes. "),
    )
    parser.add_argument(
        "--save_per_prefix",
        type=str2bool,
        default=False,
        help="Whether to save the individual shards of the dataset corresponding to each prefix length.",
    )
    parser.add_argument(
        "--overwrite_output_file",
        type=str2bool,
        default=False,
        help="Whether to overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--limit_rows",
        type=int,
        default=None,
        help="The number of rows to limit the dataset to. Useful for debugging.",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=False,
        help="Whether to print out the indexes for errors as the prefixes are generated.",
    )
    parser.add_argument(
        "--filter_length",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    ###########################################################################
    # Argument validation and conditional setting
    ###########################################################################

    # require output_dir to be specified and different from input_dir
    assert args.input_dir is not None
    assert args.output_dir is not None
    assert args.input_dir != args.output_dir, "input_dir and output_dir must be different"

    # check limit_rows
    assert (args.limit_rows is None) or (
        (args.limit_rows > 0) and isinstance(args.limit_rows, int)
    ), "limit_rows must be > 0 or None"

    # check prefix_stride
    assert (args.prefix_stride > 0) and isinstance(
        args.prefix_stride, int
    ), "prefix_stride must be > 0"

    main(args)
