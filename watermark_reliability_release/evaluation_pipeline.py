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

from types import NoneType

from typing import Union
import os
import argparse
from functools import partial
from tqdm import tqdm

import wandb
import torch
import numpy as np
import sklearn.metrics as metrics

from datasets import Dataset, Sequence
from transformers import DataCollatorWithPadding

from utils.submitit import str2bool  # better bool flag type for argparse
from utils.io import read_jsonlines, read_json, write_json, write_jsonlines
from utils.notebooks import filter_text_col_length, infer_length_column

from utils.evaluation import (
    SUPPORTED_METRICS,
    NO_CHECK_ARGS,
    ROC_TEST_STAT_SUFFIXES,
    FILTER_BY_COLUMNS,
    conditional_no_check_args,
    load_oracle_model,
    evaluate_ppl,
    load_detector,
    compute_z_scores,
    compute_windowed_z_scores,
    compute_run_len_chsqrd_stats,
    compute_repetition_diversity,
    compute_p_sp,
    compute_coherence,
    compute_mauve,
    compute_detect_retrieval,
    load_tokenizer,
    concat_rows,
)

print(f"Current huggingface cache dir: {os.environ['HF_HOME']}")

from datasets import disable_caching

disable_caching()


def main(args):
    ###########################################################################
    # Create output dir if it doesn't exist, and warn if it contains metric file
    ###########################################################################
    gen_table_w_metrics_path = f"{args.output_dir}/gen_table_w_metrics.jsonl"
    metrics_meta_path = f"{args.output_dir}/gen_table_w_metrics_meta.json"

    print(f"Output dir for this run: {args.output_dir}")
    # notify if exists
    if os.path.exists(args.output_dir):
        print(f"Output dir for this run already exists!")
        print(f"Contents: {sorted(os.listdir(args.output_dir))}")
        # warn if metrics file exists
        if os.path.exists(gen_table_w_metrics_path):
            if not args.overwrite_output_file:
                print(
                    f"WARNING: Exiting to avoid overwriting output file. "
                    f"Pass the '--overwrite_output_file' flag to ignore this check."
                )
                exit()
            else:
                print(
                    f"WARNING: Found existing generation files with metrics added at this output dir. "
                    f"Overwriting anyway :/"
                )
    else:
        # create the output dir where run artifacts are stored
        os.makedirs(args.output_dir)

    ###########################################################################
    # Parse metrics to log - ppl, zscore, etc
    ###########################################################################

    # check that all metrics are supported
    metric_support = [metric in SUPPORTED_METRICS for metric in args.evaluation_metrics]
    assert all(metric_support), (
        f"Unsupported metric '{args.evaluation_metrics[metric_support.index(False)]}' in"
        f" {args.evaluation_metrics}. Supported metrics are: {SUPPORTED_METRICS}"
    )
    # Hack check that if prefix_lengths exists then the method must be
    # detect-retrieval (for now) because other methods don't support the
    # sparse dataset with Nones all over the place
    if "prefix_lengths" in args.__dict__:
        # assert args.evaluation_metrics == [
        #     "detect-retrieval"
        # ], f"Currently, only the detect-retrieval metric supports the prefix_lengths column. "
        print(
            f"WARNING: Found prefix_lengths column assuming that this is either retireval or detectgpt"
        )

    print(f"Evaluation metrics to compute: {args.evaluation_metrics}")

    ###########################################################################
    # Load generations
    ###########################################################################
    print(f"Input dir for this run: {args.input_dir}")
    print(f"Loading previously generated outputs for evaluation via oracle model and metrics...")

    # check for the "attacked version" of the gen table first
    gen_table_meta_path = f"{args.input_dir}/gen_table_attacked_meta.json"
    gen_table_path = f"{args.input_dir}/gen_table_attacked.jsonl"
    safe_gen_table_path = f"{args.input_dir}/gen_table_attacked_safe.jsonl"
    loaded_attacked = True

    attack_variants_exist = [
        os.path.exists(gen_table_meta_path),
        os.path.exists(gen_table_path),
    ]
    if not all(attack_variants_exist):
        loaded_attacked = False
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
        if v is not None:
            joined_args.update({k: v})
        else:
            print(
                f"cmdline arg {k} is None, leaving it as the value found in the input metadata: {prev_gen_table_meta[k]}"
            )

    # check that the args used to generate the prev generations are the same as
    # the current args, for the intersection of keys
    if not args.overwrite_args:
        # update the no check args based on the current state of args
        current_no_check_args = conditional_no_check_args(
            NO_CHECK_ARGS, args.evaluation_metrics, args
        )

        for key in prev_gen_table_meta.keys():
            if key in current_no_check_args:
                continue
            assert joined_args[key] == prev_gen_table_meta[key], (
                f"failed for safety bc after merging the prev metadata with "
                f"the current cmdline args, values for '{key}' are not the same. "
                f"in metadata: {prev_gen_table_meta[key]}, passed: {cmdline_args[key]}. "
                f"Pass the '--overwrite_args' flag to ignore this check."
            )

    args = argparse.Namespace(**joined_args)
    gen_table = [ex for ex in read_jsonlines(gen_table_path)]
    if args.limit_rows == -1:
        gen_table_ds = Dataset.from_list(gen_table)
    else:
        gen_table_ds = Dataset.from_list(gen_table[: args.limit_rows])

    ###########################################################################
    # Extract the seeding scheme fine grained parameters
    ###########################################################################
    from utils.evaluation import scheme_hparam_extractor

    args.__dict__.update(scheme_hparam_extractor(args.seeding_scheme))

    print(f"seeding_scheme: {args.seeding_scheme}")
    print(f"prf_type: {args.prf_type}")
    print(f"anchored: {args.anchored}")
    print(f"context_width: {args.context_width}")
    print(f"self_salt: {args.self_salt}")

    ###########################################################################
    # Concat logic for multiple generations
    ###########################################################################

    if args.concat_rows != 0:
        assert isinstance(args.concat_rows, int), f"Invalid concat_rows arg: {args.concat_rows}. "

        # set to all rows if -1
        if args.concat_rows == -1:
            args.concat_rows = len(gen_table_ds)

        if args.shuffle_before_concat:
            print(f"Shuffling the gen table before concatenating every {args.concat_rows} rows...")
            gen_table_ds = gen_table_ds.shuffle()

        print(f"Concatenating every {args.concat_rows} rows of the gen table...")

        # we concat all cols in OUTPUT_TEXT_COLUMN_NAMES
        # and update the length col to reflect the new length
        # which means we need to tokenize the new text temporarily
        # to get the new length

        tokenizer = load_tokenizer(args)

        concat_partial = partial(concat_rows, tokenizer=tokenizer, args=args)

        # manually write a btach loop bc hf doesnt support returning fewer rows than input
        concatenated_rows = []
        for i in tqdm(range(0, len(gen_table_ds), args.concat_rows)):
            batch = gen_table_ds[i : i + args.concat_rows]
            concatenated_rows.append(concat_partial(batch))
        gen_table_concated_ds = Dataset.from_list(concatenated_rows)

        # overwrite the args.max_new_tokens to reflect the implicit new target length T
        # which is concat_rows * max_new_tokens
        args.max_new_tokens = args.concat_rows * args.max_new_tokens

        # write the dataset out in the same filename as the original
        # but check that the input dir is different from the output dir
        assert (
            args.input_dir != args.output_dir
        ), f"Input dir and output dir must be different to write out the result of concat rows."

        if loaded_attacked:
            concat_meta_path = f"{args.output_dir}/gen_table_attacked_meta.json"
            concat_gen_table_path = f"{args.output_dir}/gen_table_attacked.jsonl"
        else:
            concat_meta_path = f"{args.output_dir}/gen_table_meta.json"
            concat_gen_table_path = f"{args.output_dir}/gen_table.jsonl"

        write_json(args.__dict__, concat_meta_path, indent=4)
        gen_table_concated_lst = [ex for ex in gen_table_concated_ds]
        write_jsonlines(gen_table_concated_lst, concat_gen_table_path)
    else:
        gen_table_concated_ds = gen_table_ds

    ###########################################################################
    # Additional args setup
    ###########################################################################
    # if target_T is not specified, use max_new_tokens (which will be in the reloaded gen metadata)
    # and potentially overwritten by the concat logic above
    if args.target_T == 0:
        args.target_T = args.max_new_tokens

    # storing slurm info to allow auditing logfiles
    # note this is set after the metadata check to ignore overwriting
    args.SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
    args.SLURM_ARRAY_JOB_ID = os.getenv("SLURM_ARRAY_JOB_ID")
    args.SLURM_ARRAY_TASK_ID = os.getenv("SLURM_ARRAY_TASK_ID")

    ###########################################################################
    # Start logging, we wait to do this until after loading the generations
    # so that we can log the args used to generate them unioned with the
    # cmdline args
    ###########################################################################
    if args.wandb:
        # start a new wandb run to track this experiment, will send data to it
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
    # Perplexity (PPL) evaluation
    # NOTE: basically requires a model on gpu, or is extremely slow
    ###########################################################################
    if "ppl" in args.evaluation_metrics:
        assert args.oracle_model_name_or_path, "PPL metric requires oracle model."

        # Load the oracle model for PPL measurement
        oracle_model, oracle_tokenizer, _ = load_oracle_model(args)

        # construct the collator
        data_collator = DataCollatorWithPadding(
            tokenizer=oracle_tokenizer, padding=True, pad_to_multiple_of=8
        )

        # construct fluency/ppl partial
        evaluate_ppl_partial = partial(
            evaluate_ppl,
            oracle_model_name=args.oracle_model_name_or_path,
            oracle_model=oracle_model,
            oracle_tokenizer=oracle_tokenizer,
            data_collator=data_collator,
        )

        print(f"Computing metrics on model generations: {gen_table_concated_ds}")

        gen_table_w_ppl_ds = gen_table_concated_ds.map(
            evaluate_ppl_partial,
            batched=True,
            batch_size=args.ppl_batch_size,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

        # clear the model just for fun
        oracle_model = oracle_model.to(torch.device("cpu"))
        del oracle_model
    else:
        gen_table_w_ppl_ds = gen_table_concated_ds

    ###########################################################################
    # Cheap to load, and required for all detectors so load it first
    watermark_detector = load_detector(args)

    # Map setup for all dataset operations:
    map_setup = dict(batched=False, load_from_cache_file=False)
    ###########################################################################
    # z-score evaluation
    # NOTE: requires a gpu because if original source of watermark randomness,
    # RNG, is gpu based, then detector should be on gpu as well
    ###########################################################################
    if "z-score" in args.evaluation_metrics:
        # set up the partial
        compute_z_scores_partial = partial(
            compute_z_scores,
            watermark_detector=watermark_detector,
            args=args,
        )

        gen_table_w_zscore_ds = gen_table_w_ppl_ds.map(
            compute_z_scores_partial, **map_setup, desc="Computing z-scores"
        )

    else:
        gen_table_w_zscore_ds = gen_table_w_ppl_ds

    ###########################################################################
    # Windowed z-score evaluation
    ###########################################################################

    if "windowed-z-score" in args.evaluation_metrics:
        # set up the windowed partial
        compute_windowed_z_scores_partial = partial(
            compute_windowed_z_scores,
            watermark_detector=watermark_detector,
            args=args,
        )

        gen_table_w_windowed_zscore_ds = gen_table_w_zscore_ds.map(
            compute_windowed_z_scores_partial, **map_setup, desc="Computing windowed z-scores"
        )
    else:
        gen_table_w_windowed_zscore_ds = gen_table_w_zscore_ds

    ###########################################################################
    # run-len-chisqrd evaluation
    ###########################################################################
    if "run-len-chisqrd" in args.evaluation_metrics:
        assert "w_wm_output_green_token_mask" in gen_table_w_windowed_zscore_ds.column_names, (
            f"Currently, run-len-chisqrd metric requires the green token masks to be computed previously "
            f"by one of the z-score metrics."
        )
        # this ^ is unused currently, but we will need it to remove the assert condition above

        # set up the run len chisqrd partial
        compute_run_len_chisqrd_partial = partial(
            compute_run_len_chsqrd_stats,
            watermark_detector=watermark_detector,
            args=args,
        )

        gen_table_w_run_len_chisqrd_ds = gen_table_w_windowed_zscore_ds.map(
            compute_run_len_chisqrd_partial, **map_setup, desc="Computing runlength tests"
        )
    else:
        gen_table_w_run_len_chisqrd_ds = gen_table_w_windowed_zscore_ds

    ###########################################################################
    # Diversity and Repetition evaluation
    ###########################################################################

    if "repetition" in args.evaluation_metrics or "diversity" in args.evaluation_metrics:
        # set up the partial
        compute_repetition_partial = partial(
            compute_repetition_diversity,
            include_repetition=("repetition" in args.evaluation_metrics),
            include_diversity=("diversity" in args.evaluation_metrics),
        )

        gen_table_w_repetition_ds = gen_table_w_run_len_chisqrd_ds.map(
            compute_repetition_partial, **map_setup, desc="Computing text repetition and diversity"
        )
    else:
        gen_table_w_repetition_ds = gen_table_w_run_len_chisqrd_ds

    ###########################################################################
    # P-SP evaluation
    ###########################################################################

    if "p-sp" in args.evaluation_metrics:
        print(f"Loading the P-SP model and computing P-SP")
        gen_table_w_p_sp_ds = compute_p_sp(gen_table_w_repetition_ds)
    else:
        gen_table_w_p_sp_ds = gen_table_w_repetition_ds

    ###########################################################################
    # Coherence evaluation
    ###########################################################################

    if "coherence" in args.evaluation_metrics:
        print(f"Computing coherence")
        gen_table_w_coherence_ds = compute_coherence(gen_table_w_p_sp_ds)
    else:
        gen_table_w_coherence_ds = gen_table_w_p_sp_ds

    ###########################################################################
    # Mauve evaluation
    ###########################################################################

    if "mauve" in args.evaluation_metrics:
        print(f"Computing mauve")
        gen_table_w_mauve_ds = compute_mauve(gen_table_w_coherence_ds)
    else:
        gen_table_w_mauve_ds = gen_table_w_coherence_ds

    ###########################################################################
    # Retrieval detection
    ###########################################################################

    if "detect-retrieval" in args.evaluation_metrics:
        print(f"Computing detect retrieval")
        gen_table_w_detect_retrieval_ds = compute_detect_retrieval(gen_table_w_mauve_ds, args=args)
    else:
        gen_table_w_detect_retrieval_ds = gen_table_w_mauve_ds

    if "prefix_length" in gen_table_w_detect_retrieval_ds.features:
        if "no_wm_output_retrieval_score" in gen_table_w_detect_retrieval_ds.features:
            print("Avg scores at each prefix length for no_wm_output:")
            print(
                gen_table_w_detect_retrieval_ds.to_pandas()
                .groupby("prefix_length")["no_wm_output_retrieval_score"]
                .describe()
            )
        if "w_wm_output_retrieval_score" in gen_table_w_detect_retrieval_ds.features:
            print("Avg scores at each prefix length for w_wm_output:")
            print(
                gen_table_w_detect_retrieval_ds.to_pandas()
                .groupby("prefix_length")["w_wm_output_retrieval_score"]
                .describe()
            )
        if "w_wm_output_attacked_retrieval_score" in gen_table_w_detect_retrieval_ds.features:
            print("Avg scores at each prefix length for no_wm_output_attacked:")
            print(
                gen_table_w_detect_retrieval_ds.to_pandas()
                .groupby("prefix_length")["w_wm_output_attacked_retrieval_score"]
                .describe()
            )

    ###########################################################################
    # Detectgpt detection
    ###########################################################################
    if "detectgpt" in args.evaluation_metrics:
        assert args.evaluation_metrics == ["detectgpt"], (
            f"Detectgpt must be run separately from other metrics. "
            f"Found: {args.evaluation_metrics}. "
        )
        # check that the right score column exists
        assert any(
            ["detectgpt_score" in col for col in gen_table_w_detect_retrieval_ds.column_names]
        ), (
            f"Detectgpt metric requires the detectgpt_score column to be computed previously "
            f"but no such cols exist in this file."
        )
        print(
            f"Evaluating detectgpt by simply computing ROC-AUC metrics on the scores that already exist"
        )
        gen_table_w_metrics_ds = gen_table_w_detect_retrieval_ds

        # if we loaded an attack file, since detect gpt only outputs a baseline score col
        # and a no_wm_output score col (which is implcitly the attack col if the file was attacked)
        # we need to add the attacked score col to the dataset, and remove the no_wm score col
        if loaded_attacked:
            for suff in ["100_d", "100_z"]:
                gen_table_w_metrics_ds = gen_table_w_metrics_ds.add_column(
                    f"w_wm_output_attacked_detectgpt_score_{suff}",
                    gen_table_w_metrics_ds[f"no_wm_output_detectgpt_score_{suff}"],
                )
                gen_table_w_metrics_ds = gen_table_w_metrics_ds.remove_columns(
                    [f"no_wm_output_detectgpt_score_{suff}"]
                )
    else:
        ###########################################################################
        # Write the final dataset out to disk in jsonl format
        # with the metrics added
        ###########################################################################

        # last applied metric, NOTE which will of course change as more are added
        gen_table_w_metrics_ds = gen_table_w_detect_retrieval_ds

        # write the metadata file, which is a union of the previous metadata
        # and the current cmdline args
        write_json(args.__dict__, metrics_meta_path, indent=4)

        gen_table_w_metrics_lst = [ex for ex in gen_table_w_metrics_ds]
        write_jsonlines(gen_table_w_metrics_lst, gen_table_w_metrics_path)

    ###########################################################################
    # Log the metric series to wandb
    ###########################################################################
    # log the metrics to wandb
    if args.wandb:
        # find cols that should be logged in a table
        tabular_column_types = ["string", "bool"]
        tabular_column_names = [
            name
            for name, _ in filter(
                lambda tup: tup[1].dtype in tabular_column_types,
                gen_table_w_metrics_ds.features.items(),
            )
        ]
        # the rest should be logged as series
        series_column_names = [
            name
            for name, _ in filter(
                lambda tup: tup[1].dtype not in tabular_column_types,
                gen_table_w_metrics_ds.features.items(),
            )
        ]

        for metric_name in series_column_names:
            # summarize series metrics as mean by default
            wandb.define_metric(metric_name, summary="mean")

        if args.log_raw_series:
            # log the raw series
            for example in tqdm(
                gen_table_w_metrics_ds.remove_columns(tabular_column_names),
                desc="Logging series metrics to wandb",
            ):
                run.log(example)

        if args.log_raw_tabular:
            # log the raw tabular data
            # but also include the dataset index as a column
            series_column_names.remove("idx")
            table = wandb.Table(
                dataframe=gen_table_w_metrics_ds.remove_columns(series_column_names).to_pandas()
            )
            run.log({"output_table": table})

        ###########################################################################
        # Filter rows, then log means to wandb
        ###########################################################################
        assert (
            args.target_T - args.lower_tolerance_T
        ) >= 0, "target_T - lower_tolerance_T must be >= 0"

        target_T = args.target_T
        lower_tolerance = args.lower_tolerance_T
        upper_tolerance = args.upper_tolerance_T
        filtered_table = gen_table_w_metrics_ds.to_pandas()  # explictly convert lists

        for col in args.filter_by_columns:
            length_col_name = infer_length_column(col, filtered_table, args=args)
            filtered_table = filter_text_col_length(
                filtered_table,
                text_col_name=length_col_name,
                count_suffix="",
                upper_T=target_T + upper_tolerance,
                lower_T=target_T - lower_tolerance,
            )

        # Save filtered mean values:
        for metric_name in series_column_names:
            filtered_name = f"f_{target_T}p{upper_tolerance}m{lower_tolerance}_{metric_name}"
            try:
                run.summary[f"{filtered_name}_mean"] = filtered_table[metric_name].mean()
                run.summary[f"{filtered_name}_std"] = filtered_table[metric_name].std()
            except TypeError:
                two_dim_mean = filtered_table[metric_name].apply(np.mean).mean()

        ###########################################################################
        # Compute ROC-AUC and send to wandb
        ###########################################################################
        try:
            test_stats = args.roc_test_stat
            if isinstance(test_stats, str):
                test_stats = [test_stats]
            for test_stat in test_stats:
                for attacked in [True, False]:
                    try:
                        roc_auc, fpr, tpr, thresholds, tpr_at_X_fpr = _roc_metrics_for_wandb(
                            filtered_table, test_stat, attacked=attacked
                        )
                        run.summary[
                            f"{'attacked_' if attacked else ''}{test_stat}_roc_auc"
                        ] = roc_auc
                        run.summary[
                            f"{'attacked_' if attacked else ''}{test_stat}_tpr_at_X_fpr"
                        ] = tpr_at_X_fpr

                        # for tp, fp, thr in tqdm(
                        #     zip(tpr, fpr, thresholds), desc="Logging ROC curve"
                        # ):
                        #     run.log(
                        #         {
                        #             f"{'attacked_' if attacked else ''}{test_stat}_fpr": fp,
                        #             f"{'attacked_' if attacked else ''}{test_stat}_tpr": tp,
                        #             f"{'attacked_' if attacked else ''}thr": thr,
                        #         }
                        #     )
                        data = [[x, y] for (x, y) in zip(fpr, tpr)]
                        table = wandb.Table(data=data, columns=["fpr", "tpr"])
                        run.log(
                            {
                                f"{'attacked_' if attacked else ''}{test_stat}": wandb.plot.line(
                                    table,
                                    "fpr",
                                    "tpr",
                                    title=f"ROC ({test_stat}{',attacked' if attacked else ',clean'})",
                                )
                            }
                        )
                        print(f"Successfully logged ROC-AUC metrics for {test_stat}.")

                    except Exception as e:
                        if args.verbose:
                            print(e)
                            print(
                                f"Failed to log ROC-AUC metrics for {'attacked output' if attacked else ''} {test_stat}."
                                f"Metric probably was not computed and or attack col not present."
                            )
        except Exception as e:
            if args.verbose:
                print(f"Exception: {e}")
                print(
                    f"Failed to log ROC-AUC metrics. ",
                    f"Make sure the test statistic required for detection ({test_stat}) has been computed!",
                )

        ################################################################################
        # NOTE we do that ^^^ basic ROC logic first because it's faster
        # as well as the manual prefix lengths at T logic bc that's also faster
        ################################################################################

        # Handle z @ T but for the retrieval and detectgpt scores that are evaluated
        # manually at each prefix length.  Use groupby to compute the mean and std
        # for each prefix length for any of the feats that have retrieval_score in them,
        # then log those pairs to wandb.
        at_T_df = gen_table_w_metrics_ds.to_pandas()

        for name, feat in gen_table_w_metrics_ds.features.items():
            if "retrieval_score" in name and "prefix_length" in at_T_df.columns:
                # compute the mean and std for each prefix length
                # and log those pairs to wandb
                df_view = at_T_df.groupby("prefix_length")[name].describe()[["mean", "std"]]
                T_indices = df_view.index

                # for idx, (mean, std) in df_view.iterrows():
                #     run.log(data={f"{name}_mean": mean, f"{name}_std": std, "idx_T": idx})
                # log this triple as a table instead like the ROC curve above
                # where the first two are plotted and the third is the x axis
                data = [[x, y, z] for x, (y, z) in df_view.iterrows()]
                table = wandb.Table(data=data, columns=["idx_T", "mean", "std"])
                # compute stderr from std
                table.add_column(
                    "stderr",
                    [
                        std / np.sqrt(len(at_T_df[at_T_df["prefix_length"] == idx]))
                        for idx, std in zip(T_indices, df_view["std"])
                    ],
                )
                # first log mean
                run.log({f"{name}": wandb.plot.line(table, "idx_T", "mean", title=f"{name} mean")})
                # then log std err
                run.log(
                    {
                        f"{name}_stderr": wandb.plot.line(
                            table, "idx_T", "stderr", title=f"{name} stderr"
                        )
                    }
                )

                # also compute an AUC at each prefix len idx by treating the name col as the positives
                # and the baseline_completion_retrieval_score as the negatives
                # then log those pairs to wandb
                if name != "baseline_completion_retrieval_score":
                    pos_negs_at_T = at_T_df.groupby("prefix_length")[
                        [name, "baseline_completion_retrieval_score"]
                    ]
                    # auc_at_T = []
                    # tpr_at_X_fpr = []
                    all_aucs, all_tpr_at_X_fpr = [], []
                    for idx, sub_df in pos_negs_at_T:
                        pos = sub_df[name]
                        neg = sub_df["baseline_completion_retrieval_score"]
                        # convert to arrays and remove nans
                        pos = pos.to_numpy()[~np.isnan(pos.to_numpy())]
                        neg = neg.to_numpy()[~np.isnan(neg.to_numpy())]

                        fpr, tpr, thresholds = metrics.roc_curve(
                            np.concatenate([np.ones_like(pos), np.zeros_like(neg)]),  # labels
                            np.concatenate([pos, neg]),  # scores
                            pos_label=1,
                        )
                        auc = metrics.auc(fpr, tpr)
                        try:
                            tpr_at_X_fpr = tpr[np.where(fpr < 1e-3)[0][-1]]
                        except IndexError:
                            tpr_at_X_fpr = float("NaN")
                        all_aucs.append(auc)
                        all_tpr_at_X_fpr.append(tpr_at_X_fpr)

                        # run.log(data={f"{name}_auc_at_T": auc, "idx_T": idx})
                    # log this triple as a table instead like the AUC and tpr at X fpr below
                    # where the first two are plotted and the third is the x axis
                    data = [
                        [x, y, z] for x, (y, z) in zip(T_indices, zip(all_aucs, all_tpr_at_X_fpr))
                    ]
                    table = wandb.Table(data=data, columns=["idx_T", "aucs", "tpr_at"])
                    run.log(
                        {
                            f"{name}_aucs": wandb.plot.line(
                                table, "idx_T", "aucs", title=f"{name} aucs"
                            )
                        }
                    )
                    run.log(
                        {
                            f"{name}_tpr_at": wandb.plot.line(
                                table, "idx_T", "tpr_at", title=f"{name} tpr_at"
                            )
                        }
                    )

            elif "detectgpt_score" in name and "prefix_length" in at_T_df.columns:
                # this covers detectgpt_score_100_d and variants
                # compute the mean and std for each prefix length
                # and log those pairs to wandb
                df_view = at_T_df.groupby("prefix_length")[name].describe()[["mean", "std"]]
                T_indices = df_view.index

                # for idx, (mean, std) in df_view.iterrows():
                #     run.log(data={f"{name}_mean": mean, f"{name}_std": std, "idx_T": idx})
                # log this triple as a table instead like the ROC curve above
                # where the first two are plotted and the third is the x axis
                data = [[x, y, z] for x, (y, z) in df_view.iterrows()]
                table = wandb.Table(data=data, columns=["idx_T", "mean", "std"])

                # compute stderr from std
                table.add_column(
                    "stderr",
                    [
                        std / np.sqrt(len(at_T_df[at_T_df["prefix_length"] == idx]))
                        for idx, std in zip(T_indices, df_view["std"])
                    ],
                )
                # first log mean
                run.log({f"{name}": wandb.plot.line(table, "idx_T", "mean", title=f"{name} mean")})
                # then log std err
                run.log(
                    {
                        f"{name}_stderr": wandb.plot.line(
                            table, "idx_T", "stderr", title=f"{name} stderr"
                        )
                    }
                )

                # also compute an AUC at each prefix len idx by treating the name col as the positives
                # and the baseline_completion_retrieval_score as the negatives
                # then log those pairs to wandb
                if "baseline_completion_detectgpt_score" not in name:
                    # check which suffix this is in ["_100_d", "_100_z"]
                    # and use that to set the baseline/falst col
                    if name.endswith("_100_d"):
                        baseline_col = "baseline_completion_detectgpt_score_100_d"
                    elif name.endswith("_100_z"):
                        baseline_col = "baseline_completion_detectgpt_score_100_z"
                    pos_negs_at_T = at_T_df.groupby("prefix_length")[[name, baseline_col]]
                    # auc_at_T = []
                    # tpr_at_X_fpr = []
                    all_aucs, all_tpr_at_X_fpr = [], []
                    for idx, sub_df in pos_negs_at_T:
                        pos = sub_df[name]
                        neg = sub_df[baseline_col]
                        # convert to arrays and remove nans
                        pos = pos.to_numpy()[~np.isnan(pos.to_numpy())]
                        neg = neg.to_numpy()[~np.isnan(neg.to_numpy())]

                        fpr, tpr, thresholds = metrics.roc_curve(
                            np.concatenate([np.ones_like(pos), np.zeros_like(neg)]),  # labels
                            np.concatenate([pos, neg]),  # scores
                            pos_label=1,
                        )
                        auc = metrics.auc(fpr, tpr)
                        try:
                            tpr_at_X_fpr = tpr[np.where(fpr < 1e-3)[0][-1]]
                        except IndexError:
                            tpr_at_X_fpr = float("NaN")
                        all_aucs.append(auc)
                        all_tpr_at_X_fpr.append(tpr_at_X_fpr)

                        # run.log(data={f"{name}_auc_at_T": auc, "idx_T": idx})
                    # log this triple as a table instead like the AUC and tpr at X fpr below
                    # where the first two are plotted and the third is the x axis
                    data = [
                        [x, y, z] for x, (y, z) in zip(T_indices, zip(all_aucs, all_tpr_at_X_fpr))
                    ]
                    table = wandb.Table(data=data, columns=["idx_T", "aucs", "tpr_at"])
                    run.log(
                        {
                            f"{name}_aucs": wandb.plot.line(
                                table, "idx_T", "aucs", title=f"{name} aucs"
                            )
                        }
                    )
                    run.log(
                        {
                            f"{name}_tpr_at": wandb.plot.line(
                                table, "idx_T", "tpr_at", title=f"{name} tpr_at"
                            )
                        }
                    )

        ###########################################################################
        # Compute our @ T detection metrics and send to wandb
        ###########################################################################

        # Merge z_at_T and other sequence metrics so they can be shown in wandb:
        for name, feat in gen_table_w_metrics_ds.features.items():
            if isinstance(feat, Sequence):
                max_feat_seq_len = max([len(l) for l in gen_table_w_metrics_ds[name]])
                merging_seq = np.zeros(max_feat_seq_len)
                counts = np.zeros(max_feat_seq_len)
                proto_variance = np.zeros(max_feat_seq_len)
                for entry in gen_table_w_metrics_ds[name]:
                    len_seq = len(entry)
                    delta = entry * counts[:len_seq] - merging_seq[:len_seq]
                    # Accumulate ragged sum over entries:
                    counts[:len_seq] += 1
                    merging_seq[:len_seq] += entry[: len(merging_seq)]
                    # Compute ragged, running variance via Welford:
                    gamma = entry * counts[:len_seq] - merging_seq[:len_seq]
                    proto_variance[:len_seq] += (delta / counts[:len_seq]) * (
                        gamma / counts[:len_seq]
                    )

                mask = counts != 0
                averaged_seq = merging_seq.copy()
                averaged_seq[mask] /= counts
                averaged_seq[~mask] = float("NaN")

                seq_stderr = proto_variance.copy()
                seq_stderr[counts > 1] = np.sqrt(
                    proto_variance[counts > 1] / (counts[counts > 1] - 1)
                ) / np.sqrt(counts[counts > 1])
                seq_stderr[counts <= 1] = float("NaN")
                # for idx, (avg, stderr) in enumerate(zip(averaged_seq[mask], seq_stderr[mask])):
                #     run.log(data={f"{name}_avg": avg, f"{name}_stderr": stderr, "idx_T": idx})
                # log this triple as a table instead like the ROC curve above
                # where the first two are plotted and the third is the x axis
                data = [
                    [x, y, z]
                    for (x, y, z) in zip(
                        averaged_seq[mask], seq_stderr[mask], range(len(averaged_seq[mask]))
                    )
                ]
                table = wandb.Table(data=data, columns=["avg", "stderr", "idx_T"])

                # first plot avg
                run.log({f"{name}": wandb.plot.line(table, "idx_T", "avg", title=f"{name} avg")})
                # then plot stderr
                run.log(
                    {
                        f"{name}_stderr": wandb.plot.line(
                            table, "idx_T", "stderr", title=f"{name} stderr"
                        )
                    }
                )

        # Compute AUC_at_T
        # For now we'll just do a dumb loop over scipy.roc_curve, but this could be batched
        test_stats = args.roc_test_stat
        if isinstance(test_stats, str):
            test_stats = [test_stats]

        for test_stat in test_stats:
            for attacked in [True, False]:
                base_col = f"baseline_completion_{test_stat}_at_T"
                w_wm_col = f"w_wm_output{'_attacked' if attacked else ''}_{test_stat}_at_T"
                name = f"w_wm{'_attacked' if attacked else ''}_{test_stat}_at_T"

                if w_wm_col in gen_table_w_metrics_ds.features.keys():  # metric was computed
                    print(f"Computing AUC at T for {name}.")
                    max_length = min(
                        max([len(l) for l in gen_table_w_metrics_ds[base_col]]),
                        max([len(l) for l in gen_table_w_metrics_ds[w_wm_col]]),
                    )

                    all_aucs, all_tpr_at_X_fpr = [], []
                    for T in range(1, max_length):
                        w_wm_stats = np.array(
                            [t[T] for t in gen_table_w_metrics_ds[w_wm_col] if len(t) > T]
                        )

                        baseline_stats = np.array(
                            [t[T] for t in gen_table_w_metrics_ds[base_col] if len(t) > T]
                        )[: len(w_wm_stats)]
                        all_scores = np.concatenate([baseline_stats, w_wm_stats])

                        baseline_labels = np.zeros_like(baseline_stats)
                        attacked_labels = np.ones_like(w_wm_stats)
                        all_labels = np.concatenate([baseline_labels, attacked_labels])

                        if len(np.unique(all_labels)) < 2:
                            roc_auc = float("NaN")
                            tpr_at_X_fpr = float("NaN")
                        else:
                            fpr, tpr, thresholds = metrics.roc_curve(
                                all_labels, all_scores, pos_label=1
                            )
                            roc_auc = metrics.auc(fpr, tpr)
                            try:
                                tpr_at_X_fpr = tpr[np.where(fpr < 1e-3)[0][-1]]
                            except IndexError:
                                tpr_at_X_fpr = float("NaN")

                        all_aucs.append(roc_auc)
                        all_tpr_at_X_fpr.append(tpr_at_X_fpr)
                    # for idx, (aucs, tpr_at) in enumerate(zip(all_aucs, all_tpr_at_X_fpr)):
                    #     run.log(data={f"{name}_aucs": aucs, f"{name}_tpr_at": tpr_at, "idx_T": idx})
                    # log these two separately using a table
                    data = [
                        [x, y, z]
                        for (x, y, z) in zip(all_aucs, all_tpr_at_X_fpr, range(len(all_aucs)))
                    ]
                    table = wandb.Table(data=data, columns=["aucs", "tpr_at", "idx_T"])
                    run.log(
                        {
                            f"{name}_aucs": wandb.plot.line(
                                table, "idx_T", "aucs", title=f"{name} aucs"
                            )
                        }
                    )
                    run.log(
                        {
                            f"{name}_tpr_at": wandb.plot.line(
                                table, "idx_T", "tpr_at", title=f"{name} tpr_at"
                            )
                        }
                    )

        # finish the wandb run
        run.finish()

    return


def _roc_metrics_for_wandb(
    gen_table_ds, test_stat="z_score", prefix="", attacked=False, remove_nan=True
):
    # In theory, we actually should be filtering the attacked column too, but we know these
    # end up very short sometimes. So, to make sure the logic works, we just
    # filter for any rows where the test metrics are NaN and note the damage

    baseline_col_name = f"{prefix}baseline_completion_{test_stat}"
    if "retrieval" in test_stat:
        if attacked:
            w_wm_col_name = f"{prefix}w_wm_output_attacked_retrieval_score"
        else:
            w_wm_col_name = f"{prefix}{args.retrieval_db_column}_retrieval_score"
    elif "detectgpt" in test_stat:
        if attacked:
            w_wm_col_name = f"{prefix}w_wm_output_attacked_{test_stat}"
        else:
            w_wm_col_name = f"{prefix}no_wm_output_{test_stat}"
    else:
        w_wm_col_name = f"{prefix}w_wm_output{'_attacked' if attacked else ''}_{test_stat}"

    # drop nans in either column
    if remove_nan:
        orig_length = len(gen_table_ds)
        gen_table_ds = gen_table_ds.dropna(subset=[baseline_col_name, w_wm_col_name])
        if orig_length != len(gen_table_ds):
            print(
                f"NOTE: During ROC calculation, dropped {orig_length - len(gen_table_ds)} rows due to NaNs in {baseline_col_name} or {w_wm_col_name}"
            )

    baseline_stats = gen_table_ds[baseline_col_name].values
    w_wm_stats = gen_table_ds[w_wm_col_name].values
    all_scores = np.concatenate([baseline_stats, w_wm_stats])

    baseline_labels = np.zeros_like(baseline_stats)
    attacked_labels = np.ones_like(w_wm_stats)
    all_labels = np.concatenate([baseline_labels, attacked_labels])

    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    try:
        tpr_at_X_fpr = tpr[np.where(fpr < 1e-3)[0][-1]]
    except IndexError:
        tpr_at_X_fpr = float("NaN")
    return roc_auc, fpr, tpr, thresholds, tpr_at_X_fpr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation pipeline for watermark detection")
    parser.add_argument(
        "--evaluation_metrics",
        type=str,
        default="all",
        help="Comma separated list of columns to remove from the dataset before generation.",
    )
    parser.add_argument(
        "--compute_scores_at_T",
        type=str2bool,
        default=True,
        help="Whether to compute (applicable) metrics at each T index in the output/text columns.",
    )
    parser.add_argument(
        "--overwrite_args",
        type=str2bool,
        default=False,
        help="Whether to overwrite the shared args in the metadata file with the current, runtime args.",
    )
    parser.add_argument(
        "--oracle_model_name_or_path",
        type=str,
        default="facebook/opt-6.7b",
        help="Oracle model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=None,
        help=(
            "Whether to run model (for ppl) in float16 precsion, note, will overwrite error as a reminder that "
            "generation was run in other mode, even though there's no hard requirement that these match."
        ),
    )
    parser.add_argument(
        "--ppl_batch_size",
        type=int,
        default=1,
        help="Batch size for ppl eval.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=Union[str, NoneType],
        default=None,
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=Union[float, NoneType],
        default=None,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--normalizers",
        type=Union[str, NoneType],
        default=None,
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_ngrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--return_green_token_mask",
        type=str2bool,
        default=True,
        help="Whether to return the mask marking which tokens are green from the watermark detector.",
    )
    parser.add_argument(
        "--window_settings",
        type=str,
        default="20,40,max",  # can also be "20" or "20,40,max"
        help="Comma separated list of window sizes to use for watermark detection. Only used if 'windowed-z-score' is in the evaluation metrics list.",
    )
    parser.add_argument(
        "--run_len_chisqrd_variant",
        type=str,
        default="F_succ_T_runs",
        choices=["F_succ_T_runs", "T_and_F_runs"],
        help="The variant of the run length test to use for watermark detection.",
    )
    parser.add_argument(
        "--run_len_chisqrd_bin_spec",
        type=str,
        default="max_plus_1",
        choices=["max", "max_plus_1"],
        help="The binning specification to use for the run length test.",
    )
    parser.add_argument(
        "--run_len_chisqrd_mask_zeros",
        type=str2bool,
        default=True,
        help="Whether to mask zeros in the run length test.",
    )
    parser.add_argument(
        "--run_len_chisqrd_mask_leading_bins",
        type=int,
        default=0,
        help="The number of leading bins to mask in the run length test.",
    )
    parser.add_argument(
        "--run_len_chisqrd_lambda",
        type=str,
        default="pearson",
        choices=["pearson", "g_test", "cressie_read"],
        help="The lambda_ param to use for the run length test.",
    )
    parser.add_argument(
        "--retrieval_technique",
        type=str,
        default="bm25",
        choices=["bm25", "sim"],
        help="The retrieval technique to use for retrieval detection.",
    )
    parser.add_argument(
        "--retrieval_db_column",
        type=str,
        default="no_wm_output",
        choices=["w_wm_output", "no_wm_output"],
        help="The column to populate the db/index with use for retrieval detection.",
    )
    parser.add_argument(
        "--retrieval_db_load_all_prefixes",
        type=str2bool,
        default=False,
        help="Whether to load all prefixes into the retrieval db, or just the longest for each unique entry.",
    )
    parser.add_argument(
        "--roc_test_stat",
        type=str,
        default="all",
        help="The comma separated list of test statistics to use for the ROC-AUC metric.",
    )
    parser.add_argument(
        "--target_T",
        type=int,
        default=0,
        help="The target generation length to use when dropping rows before ROC-AUC evaluation.",
    )
    parser.add_argument(
        "--lower_tolerance_T",
        type=int,
        default=25,
        help="The lower tolerance to use when dropping rows before ROC-AUC evaluation.",
    )
    parser.add_argument(
        "--upper_tolerance_T",
        type=int,
        default=25,
        help="The upper tolerance to use when dropping rows before ROC-AUC evaluation.",
    )
    parser.add_argument(
        "--filter_by_columns",
        type=str,
        default="all",
        help="The comma separated list of columns to filter by before ROC-AUC evaluation.",
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
        default="",
        help="The unique name for the run.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./input",
        help="The directory containing the input files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help=(
            "The directory in which to write out the dataset after adding the metrics. "
            "If not specified, will use the input_dir. Note, if the output_dir already "
            "contains the metric-enriched file, it will be overwritten :/"
        ),
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
        default=-1,
        help="The number of rows to limit the dataset to. Useful for debugging.",
    )
    parser.add_argument(
        "--concat_rows",
        type=int,
        default=0,
        help="The number of rows to concatenate into a single row. Result is a mangled dataset, be careful",
    )
    parser.add_argument(
        "--shuffle_before_concat",
        type=str2bool,
        default=False,
        help="Whether to shuffle the dataset before concatenating rows.",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=None,
        help="Whether to verbosely print things here and there.",
    )
    parser.add_argument(
        "--log_raw_series",
        type=str2bool,
        default=True,
        help="Whether to log the raw series metric data to wandb.",
    )
    parser.add_argument(
        "--log_raw_tabular",
        type=str2bool,
        default=True,
        help="Whether to log the raw tabular metric data to wandb.",
    )
    args = parser.parse_args()

    ###########################################################################
    # Argument validation and conditional setting
    ###########################################################################

    # convert evaluation metrics to list
    assert args.evaluation_metrics, "evaluation_metrics list must be specified"
    args.evaluation_metrics = args.evaluation_metrics.split(",")

    if args.evaluation_metrics == ["all"]:
        all_metrics = SUPPORTED_METRICS
        all_metrics.remove("ppl")  # by default not running this anymore
        all_metrics.remove("detectgpt")  # can't run this with other metrics
        args.evaluation_metrics = all_metrics
    if args.evaluation_metrics == ["all_w_ppl"]:
        args.evaluation_metrics = SUPPORTED_METRICS

    # if no output dir specified, use the input dir
    if args.output_dir == "":
        args.output_dir = args.input_dir

    # check limit_rows
    assert (args.limit_rows == -1) or (
        (args.limit_rows > 0) and isinstance(args.limit_rows, int)
    ), "limit_rows must be -1 or > 0"

    # convert normalizers to list
    if args.normalizers:
        args.normalizers = args.normalizers.split(",")
    else:
        args.normalizers = []

    # convert roc_test_stat to list
    args.roc_test_stat = args.roc_test_stat.split(",")

    if args.roc_test_stat == ["all"]:
        args.roc_test_stat = ROC_TEST_STAT_SUFFIXES

    # convert filter_by_columns to list
    args.filter_by_columns = args.filter_by_columns.split(",")
    if args.filter_by_columns == ["all"]:
        args.filter_by_columns = FILTER_BY_COLUMNS

    # split wandb tags
    if args.wandb_tags != "":
        args.wandb_tags = args.wandb_tags.split(",")
    else:
        args.wandb_tags = []

    # split window settings
    args.window_settings = args.window_settings.split(",")

    main(args)
