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

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from utils.generation import tokenize_and_truncate, collate_batch
from metrics.repetition_diversity import (
    measure_repetition_and_diversity,
    dummy_rep_div_result,
)
from metrics.p_sp import evaluate_p_sp
from metrics.detect_retrieval import detect_retrieval
from metrics.coherence import get_coherence_score
from metrics.mauve import get_mauve_score
from utils.hypothesis_testing import (
    chi_squared_runs_test,
    F_succ_T_runs_dummy_dict_w_bins,
    F_succ_T_runs_dummy_dict_no_bins,
    T_and_F_runs_dummy_dict_w_bins,
    T_and_F_runs_dummy_dict_no_bins,
)

from watermark_processor import WatermarkDetector

# These areguments are ignored when doing checks between meta file and cmdline args
NO_CHECK_ARGS = [
    "evaluation_metrics",
    "verbose",
    "wandb",
    "wandb_entity",
    "input_dir",
    "output_dir",
    "run_name",
    "overwrite_output_file",
    "overwrite_args",
    "limit_rows",
    "concat_rows",
    "max_prefix_length",
]


def conditional_no_check_args(no_check_args, evaluation_metrics, args):
    if "ppl" not in evaluation_metrics:
        no_check_args.append("oracle_model_name_or_path")
        no_check_args.append("load_fp16")
        no_check_args.append("ppl_batch_size")

    return no_check_args


# Series of configuration variables for the evaluation script
# These are the metrics we support
SUPPORTED_METRICS = [
    "z-score",
    "windowed-z-score",
    "run-len-chisqrd",
    "ppl",
    "diversity",
    "repetition",
    "p-sp",
    "coherence",
    "mauve",
    "detect-retrieval",
    "detectgpt",
]

# These are the output text columns we want to compute metrics on
OUTPUT_TEXT_COLUMN_NAMES = [
    "baseline_completion",
    "no_wm_output",
    "w_wm_output",
    "w_wm_output_attacked",
]

# etc for other evaluation types
ZSCORE_TEXT_COLUMN_NAMES = OUTPUT_TEXT_COLUMN_NAMES
RUN_LEN_CHISQRD_TEXT_COLUMN_NAMES = OUTPUT_TEXT_COLUMN_NAMES
REPETITION_TEXT_COLUMN_NAMES = OUTPUT_TEXT_COLUMN_NAMES
# note the convention of including the input as 0th column
COHERENCE_TEXT_COLUMN_NAMES = ["truncated_input"] + OUTPUT_TEXT_COLUMN_NAMES

# These are the column pairs we want to compute p-sp for
OUTPUT_TEXT_PAIR_COLUMN_NAMES = [
    ["baseline_completion", "no_wm_output"],
    ["baseline_completion", "w_wm_output"],
    ["baseline_completion", "w_wm_output_attacked"],
    ["no_wm_output", "w_wm_output"],
    ["w_wm_output", "w_wm_output_attacked"],
]

P_SP_TEXT_PAIR_COLUMN_NAMES = OUTPUT_TEXT_PAIR_COLUMN_NAMES
MAUVE_TEXT_PAIR_COLUMN_NAMES = OUTPUT_TEXT_PAIR_COLUMN_NAMES


ROC_TEST_STAT_SUFFIXES = [
    "z_score",
    "win20-1_z_score",
    "win40-1_z_score",
    "winmax-1_z_score",
    "run_len_chisqrd_statistic",
    "retrieval_score",
    "detectgpt_score_100_z",
    "detectgpt_score_100_d",
]

FILTER_BY_COLUMNS = ["baseline_completion", "no_wm_output", "w_wm_output"]


def concat_rows(examples, tokenizer=None, args=None):
    # concat the rows (there will be k rows per example)
    # just joining the strings by a space
    for col_name in examples.keys():
        if col_name in OUTPUT_TEXT_COLUMN_NAMES:
            examples[col_name] = " ".join(examples[col_name])
        else:
            # # check that all other columns have len args.concat_rows
            # if len(examples[col_name]) != args.concat_rows:
            #     # append None to the col to make it the right length
            #     examples[col_name] = examples[col_name] + [None] * (
            #         args.concat_rows - len(examples[col_name])
            #     )
            # EH for now just set them to be the first element of their respective column
            # quite mangled...
            examples[col_name] = examples[col_name][0]

    # Now, update the lengths
    for col_name in OUTPUT_TEXT_COLUMN_NAMES:
        if col_name in examples:
            examples[f"{col_name}_length"] = len(
                tokenizer(examples[col_name], add_special_tokens=False)["input_ids"]
            )
    return examples


def load_tokenizer(args):
    model_name = args.model_name_or_path
    print(f"Loading tokenizer for: {model_name}")
    if "llama" in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = 0  # unk
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def load_detector(args):
    if "llama" in args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.pad_token_id = 0  # unk
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    device = "cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"

    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        seeding_scheme=args.seeding_scheme,
        device=device,
        tokenizer=tokenizer,
        z_threshold=args.detection_z_threshold,
        normalizers=args.normalizers,
        ignore_repeated_ngrams=args.ignore_repeated_ngrams,
    )

    return watermark_detector


def compute_z_score(
    example,
    text_column_name=None,
    watermark_detector=None,
    args=None,
    window_size=None,
    window_stride=None,
):
    # for now, don't get the green token mask
    # if we're using normalizers
    return_green_token_mask = args.return_green_token_mask
    if args.normalizers != []:
        return_green_token_mask = None

    input_text = example[text_column_name]
    error = False
    if input_text == "":
        error = True
    else:
        try:
            score_dict = watermark_detector.detect(
                input_text,
                window_size=window_size,
                window_stride=window_stride,
                return_green_token_mask=return_green_token_mask,
                return_prediction=False,  # this conversion to "decision" only desired in demo context
                convert_to_float=True,  # this helps with integrity under NaNs
                return_z_at_T=args.compute_scores_at_T,
            )
        except Exception as e:
            print(e)
            error = True
    if error:
        problem_text = f"'{input_text[:40]} {'[...]' if len(input_text) > 40 else ''}'"
        if args.verbose:
            print(
                f"{(f'Windowed({window_size})' if window_size else '')} Detection error on text: {problem_text}"
            )
        # "Error string too short to compute metrics"
        score_dict = watermark_detector.dummy_detect(
            return_prediction=False,
            return_green_token_mask=return_green_token_mask,
            return_z_at_T=args.compute_scores_at_T,
        )

    # current detect logic causes issues bc it only reports this sometimes
    score_dict.pop("confidence", None)

    # replace every key name in score dict with the text_column_name + key name
    # and then add them to the example dict
    score_dict = {
        text_column_name
        + (f"_win{window_size}-{window_stride}" if window_size else "")
        + "_"
        + k: v
        for k, v in score_dict.items()
    }
    example.update(score_dict)
    return example


def compute_z_scores(example, watermark_detector=None, args=None):
    # this just iterates the z-score function over the columns we want to compute z-scores for
    for col_name in ZSCORE_TEXT_COLUMN_NAMES:
        if col_name in example:
            example = compute_z_score(
                example, text_column_name=col_name, watermark_detector=watermark_detector, args=args
            )
    return example


def compute_windowed_z_scores(example, watermark_detector=None, args=None):
    # this iterates the z-score function over the columns we want to compute z-scores for
    for col_name in ZSCORE_TEXT_COLUMN_NAMES:
        if col_name in example:
            for window_size in args.window_settings:
                example = compute_z_score(
                    example,
                    text_column_name=col_name,
                    watermark_detector=watermark_detector,
                    args=args,
                    window_size=window_size,
                    window_stride=1,
                )
    return example


def compute_run_len_chisqrd_stat(
    example,
    text_column_name=None,
    bool_arr_suffix=None,
    bool_arr=None,
    watermark_detector=None,  # unused under the "z-score required to be run first" assumption
    args=None,
    force_error=False,
):
    if bool_arr is not None:
        bool_array = bool_arr
    else:
        bool_array_col_name = text_column_name + bool_arr_suffix
        bool_array = example[bool_array_col_name]
    if isinstance(bool_array, list):
        bool_array = np.array(bool_array)

    run_len_kwargs = dict(
        bool_arr=bool_array,
        succ_prob=1 - args.gamma,  # this applies for both variants
        variant=args.run_len_chisqrd_variant,
        bin_spec=args.run_len_chisqrd_bin_spec,
        verbose=False,  # likely never in this context
        invert_bools=False,  # legacy
        return_bin_counts=False,  # debugging only, may not work currently
        mask_zeros=args.run_len_chisqrd_mask_zeros,
        mask_leading_bins=args.run_len_chisqrd_mask_leading_bins,
        diy=False,  # legacy
        lambda_=args.run_len_chisqrd_lambda,
        return_dict=True,  # always in this context
    )

    error = True if force_error else False
    try:
        score_dict = chi_squared_runs_test(**run_len_kwargs)
    except Exception as e:
        print(e)
        error = True
    if error:
        print(f"Run length test error, got: '{bool_array}'")
        if run_len_kwargs["variant"] == "F_succ_T_runs":
            if run_len_kwargs["return_bin_counts"]:
                score_dict = F_succ_T_runs_dummy_dict_w_bins
            else:
                score_dict = F_succ_T_runs_dummy_dict_no_bins
        elif run_len_kwargs["variant"] == "T_and_F_runs":
            if run_len_kwargs["return_bin_counts"]:
                score_dict = T_and_F_runs_dummy_dict_w_bins
            else:
                score_dict = T_and_F_runs_dummy_dict_no_bins
        else:
            raise ValueError("Unknown run length test variant and return_bin_counts setting")

    # replace every key name in score dict with the text_column_name + key name
    # and then add them to the example dict
    score_dict = {text_column_name + "_run_len_chisqrd_" + k: v for k, v in score_dict.items()}
    example.update(score_dict)

    return example


def compute_run_len_chsqrd_stats(
    example,
    watermark_detector=None,
    args=None,
    bool_arr_suffix="_green_token_mask",
    score_suffix="_run_len_chisqrd_statistic",
):
    # this just iterates the run_len_chisqrd function over the columns we want to compute stats for
    for col_name in RUN_LEN_CHISQRD_TEXT_COLUMN_NAMES:
        if col_name in example:
            if args.compute_scores_at_T:
                full_bool_arr = example[f"{col_name}{bool_arr_suffix}"]
                len_sequence = len(full_bool_arr)
                if len_sequence < 1:
                    force_error = True
                    full_bool_arr = [None]  # to cause loop to happen
                    len_sequence = 1
                else:
                    force_error = False
                stats_at_T = []
                for t in range(1, len_sequence + 1):
                    bool_arr = full_bool_arr[:t]
                    example = compute_run_len_chisqrd_stat(
                        example,
                        bool_arr=bool_arr,  # this overrides the normal access of the bool_arr
                        text_column_name=col_name,
                        bool_arr_suffix=bool_arr_suffix,
                        watermark_detector=watermark_detector,
                        args=args,
                        force_error=force_error,
                    )
                    stats_at_T.append(example[f"{col_name}{score_suffix}"])
                example[f"{col_name}{score_suffix}_at_T"] = stats_at_T
            else:
                example = compute_run_len_chisqrd_stat(
                    example,
                    text_column_name=col_name,
                    bool_arr_suffix=bool_arr_suffix,
                    watermark_detector=watermark_detector,
                    args=args,
                )
    return example


def load_oracle_model(args):
    oracle_model_name = args.oracle_model_name_or_path
    print(f"Loading oracle model: {oracle_model_name}")
    if args.load_fp16:
        oracle_model = AutoModelForCausalLM.from_pretrained(
            oracle_model_name, torch_dtype=torch.float16, device_map="auto"
        )
    else:
        oracle_model = AutoModelForCausalLM.from_pretrained(oracle_model_name)
    if "llama" in oracle_model_name:
        oracle_tokenizer = LlamaTokenizer.from_pretrained(oracle_model_name)
        oracle_model.config.pad_token_id = oracle_tokenizer.pad_token_id = 0  # unk
        oracle_model.config.bos_token_id = 1
        oracle_model.config.eos_token_id = 2
    else:
        oracle_tokenizer = AutoTokenizer.from_pretrained(oracle_model_name)
    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not args.load_fp16:
            oracle_model = oracle_model.to(device)
    else:
        device = "cpu"
    oracle_model.eval()

    return oracle_model, oracle_tokenizer, device


from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast


def opt_unpooled_loss(logits, labels, model):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))
    loss = loss.reshape(shift_logits.shape[:-1])
    # compute the mean for each elm in batch where the label is not pad
    # we assume the losses are zero for pad indices
    loss = torch.sum(loss, dim=-1) / torch.sum(shift_labels != -100, dim=-1)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
    )


UNPOOL_FN_TABLE = {
    "opt": opt_unpooled_loss,
}


def get_unpool_fn(model_name):
    if "opt" in model_name:
        return UNPOOL_FN_TABLE["opt"]
    else:
        raise NotImplementedError(f"unpooling function not implemented for {model_name}")


def compute_ppl_batch(
    prefix_and_output_text=None,
    output_text=None,
    oracle_model_name=None,
    oracle_model=None,
    oracle_tokenizer=None,
    data_collator=None,
):
    inputs = []
    labels = []
    for idx in range(len(prefix_and_output_text)):
        tokd_prefix = tokenize_and_truncate(
            {"text": prefix_and_output_text[idx]},
            completion_length=0,
            hf_model_name=oracle_model_name,
            tokenizer=oracle_tokenizer,
            truncate_left=True,  # we add this to cover if the generation is longer than the oracle's max length
            model_max_length=oracle_model.config.max_position_embeddings,
        )["input_ids"]

        # if only want to score the "generation" part we need the suffix tokenization length
        tokd_suffix = tokenize_and_truncate(
            {"text": output_text[idx]},
            completion_length=0,
            hf_model_name=oracle_model_name,
            tokenizer=oracle_tokenizer,
        )["input_ids"]

        tokd_labels = tokd_prefix.clone().detach()
        tokd_labels[:, : tokd_labels.shape[1] - tokd_suffix.shape[1] + 1] = -100

        inputs.append(tokd_prefix)
        labels.append(tokd_labels)

    inputs = collate_batch(input_ids=inputs, collator=data_collator).to(oracle_model.device)
    labels = collate_batch(input_ids=labels, collator=data_collator).to(oracle_model.device)

    labels[labels == oracle_tokenizer.pad_token_id] = -100  # mask out pad tokens for loss

    with torch.no_grad():
        pooled_outputs = oracle_model(input_ids=inputs, labels=labels)

        outputs = get_unpool_fn(oracle_model_name)(pooled_outputs.logits, labels, oracle_model)
        loss = (
            outputs.loss
        )  # avg CE loss all sequence positions (except where labels -100, i.e. pad)
        # ppl = torch.tensor(math.exp(loss))
        ppl = torch.exp(loss)

    return loss.tolist(), ppl.tolist()


def evaluate_ppl(
    examples: dict,
    oracle_model_name=None,
    oracle_model=None,
    oracle_tokenizer=None,
    data_collator=None,
):
    inputs_plus_baseline_outputs = []
    baseline_outputs = []
    inputs_plus_no_wm_outputs = []
    no_wm_outputs = []
    inputs_plus_w_wm_outputs = []
    w_wm_outputs = []
    inputs_plus_w_wm_output_attackeds = []
    w_wm_output_attackeds = []

    for idx in range(len(examples["truncated_input"])):
        # pull out the required fields from the pipeline results
        inputs_plus_baseline_output = (
            f"{examples['truncated_input'][idx]}{examples['baseline_completion'][idx]}"
        )
        baseline_output = f"{examples['baseline_completion'][idx]}"

        inputs_plus_no_wm_output = (
            f"{examples['truncated_input'][idx]}{examples['no_wm_output'][idx]}"
        )
        no_wm_output = f"{examples['no_wm_output'][idx]}"

        inputs_plus_w_wm_output = (
            f"{examples['truncated_input'][idx]}{examples['w_wm_output'][idx]}"
        )
        w_wm_output = f"{examples['w_wm_output'][idx]}"

        if "w_wm_output_attacked" in examples:
            inputs_plus_w_wm_output_attacked = (
                f"{examples['truncated_input'][idx]}{examples['w_wm_output_attacked'][idx]}"
            )
            w_wm_output_attacked = f"{examples['w_wm_output_attacked'][idx]}"

        # add to lists
        inputs_plus_baseline_outputs.append(inputs_plus_baseline_output)
        baseline_outputs.append(baseline_output)
        inputs_plus_no_wm_outputs.append(inputs_plus_no_wm_output)
        no_wm_outputs.append(no_wm_output)
        inputs_plus_w_wm_outputs.append(inputs_plus_w_wm_output)
        w_wm_outputs.append(w_wm_output)
        if "w_wm_output_attacked" in examples:
            inputs_plus_w_wm_output_attackeds.append(inputs_plus_w_wm_output_attacked)
            w_wm_output_attackeds.append(w_wm_output_attacked)

    # add metrics
    loss, ppl = compute_ppl_batch(
        inputs_plus_baseline_outputs,
        baseline_outputs,
        oracle_model_name,
        oracle_model,
        oracle_tokenizer,
        data_collator=data_collator,
    )
    examples["baseline_completion_loss"] = loss
    examples["baseline_completion_ppl"] = ppl

    loss, ppl = compute_ppl_batch(
        inputs_plus_no_wm_outputs,
        no_wm_outputs,
        oracle_model_name,
        oracle_model,
        oracle_tokenizer,
        data_collator=data_collator,
    )
    examples["no_wm_output_loss"] = loss
    examples["no_wm_output_ppl"] = ppl

    loss, ppl = compute_ppl_batch(
        inputs_plus_w_wm_outputs,
        w_wm_outputs,
        oracle_model_name,
        oracle_model,
        oracle_tokenizer,
        data_collator=data_collator,
    )
    examples["w_wm_output_loss"] = loss
    examples["w_wm_output_ppl"] = ppl

    if "w_wm_output_attacked" in examples:
        loss, ppl = compute_ppl_batch(
            inputs_plus_w_wm_output_attackeds,
            w_wm_output_attackeds,
            oracle_model_name,
            oracle_model,
            oracle_tokenizer,
            data_collator=data_collator,
        )
        examples["w_wm_output_attacked_loss"] = loss
        examples["w_wm_output_attacked_ppl"] = ppl

    return examples


def compute_repetition_diversity(example, include_repetition=False, include_diversity=False):
    for col_name in REPETITION_TEXT_COLUMN_NAMES:
        if col_name in example:
            try:
                results_tuple = measure_repetition_and_diversity(example[col_name])
            except Exception as e:
                print(
                    f"Error for '{col_name}' computing repetition and diversity on text: '{example[col_name]}'\nError:{e}"
                )
                results_tuple = dummy_rep_div_result

            if include_repetition:
                # returns pred_seq_2, pred_seq_3, pred_seq_4, pred_div
                # add each key from the result tuple to the example, prepending the col_name
                metrics_dict = {f"{col_name}_{key}": value for key, value in results_tuple.items()}
                example.update(metrics_dict)
            if include_diversity:
                # returns diversity only
                example[f"{col_name}_diversity"] = results_tuple["diversity"]
                example[f"{col_name}_log_diversity"] = results_tuple["log_diversity"]
    return example


def compute_p_sp(dataset):
    for column_pair in P_SP_TEXT_PAIR_COLUMN_NAMES:
        if column_pair[0] in dataset.features and column_pair[1] in dataset.features:
            p_sp_scores = evaluate_p_sp(dataset[column_pair[0]], dataset[column_pair[1]])
            if f"{column_pair[0]}_vs_{column_pair[1]}_p_sp" in dataset.features:
                print(
                    f"WARNING: Removing existing {column_pair[0]}_vs_{column_pair[1]}_p_sp column because it was already present"
                )
                dataset = dataset.remove_columns([f"{column_pair[0]}_vs_{column_pair[1]}_p_sp"])
            dataset = dataset.add_column(f"{column_pair[0]}_vs_{column_pair[1]}_p_sp", p_sp_scores)
    return dataset


def compute_mauve(dataset):
    """
    The current convention is to repeat the score for all rows in the dataset
    under the assumption that the final score will be retreived via
    a groupby + take(1) operation or similar (even a `mean` would be fine)
    """
    for column_pair in MAUVE_TEXT_PAIR_COLUMN_NAMES:
        if column_pair[0] in dataset.features and column_pair[1] in dataset.features:
            mauve_score = get_mauve_score(dataset[column_pair[0]], dataset[column_pair[1]])
            if f"{column_pair[0]}_vs_{column_pair[1]}_mauve" in dataset.features:
                print(
                    f"WARNING: Removing existing {column_pair[0]}_vs_{column_pair[1]}_mauve column because it was already present"
                )
                dataset = dataset.remove_columns([f"{column_pair[0]}_vs_{column_pair[1]}_mauve"])
            dataset = dataset.add_column(
                f"{column_pair[0]}_vs_{column_pair[1]}_mauve", [mauve_score] * len(dataset)
            )
    return dataset


def compute_coherence(dataset):
    """
    Assumes the first column is the prefix or prompt to the model
    and the current convention is to repeat the score for all rows in the dataset
    under the assumption that the final score will be retreived via
    a groupby + take(1) operation or similar (even a `mean` would be fine)
    """
    prefix_column = dataset[COHERENCE_TEXT_COLUMN_NAMES[0]]
    for generated_text_column in COHERENCE_TEXT_COLUMN_NAMES[1:]:
        if generated_text_column in dataset.features:
            coherence_score = get_coherence_score(prefix_column, dataset[generated_text_column])
            if f"{generated_text_column}_coherence" in dataset.features:
                print(
                    f"WARNING: Removing existing {generated_text_column}_coherence column because it was already present"
                )
                dataset = dataset.remove_columns([f"{generated_text_column}_coherence"])
            dataset = dataset.add_column(
                f"{generated_text_column}_coherence", [coherence_score] * len(dataset)
            )
    return dataset


def compute_detect_retrieval(dataset, args=None):
    # if we don't have the attacked column,
    # then mock it using the w_wm_output, just means the two score cols will be the same
    # and we'll need to delete it after
    was_real_attacked_ds = True
    if "w_wm_output_attacked" not in dataset.features:
        # were faking it
        was_real_attacked_ds = False
        dataset = dataset.add_column("w_wm_output_attacked", dataset[args.retrieval_db_column])
        dataset = dataset.add_column(
            "w_wm_output_attacked_length", dataset[f"{args.retrieval_db_column}_length"]
        )

    human_detect, paraphrase_detect, generation_detect = detect_retrieval(dataset, args=args)

    if f"baseline_completion_retrieval_score" in dataset.features:
        print(
            f"WARNING: Removing existing baseline_completion_retrieval_score column because it was already present"
        )
        dataset = dataset.remove_columns(["baseline_completion_retrieval_score"])
    dataset = dataset.add_column(f"baseline_completion_retrieval_score", human_detect)

    if f"{args.retrieval_db_column}_retrieval_score" in dataset.features:
        print(
            f"WARNING: Removing existing {args.retrieval_db_column}_retrieval_score column because it was already present"
        )
        dataset = dataset.remove_columns([f"{args.retrieval_db_column}_retrieval_score"])
    dataset = dataset.add_column(f"{args.retrieval_db_column}_retrieval_score", generation_detect)

    if was_real_attacked_ds:
        if f"w_wm_output_attacked_retrieval_score" in dataset.features:
            print(
                f"WARNING: Removing existing w_wm_output_attacked_retrieval_score column because it was already present"
            )
            dataset = dataset.remove_columns(["w_wm_output_attacked_retrieval_score"])
        dataset = dataset.add_column(f"w_wm_output_attacked_retrieval_score", paraphrase_detect)
        # else this is a dummy column, so delete it
    else:
        # sanity check that the scores are the same for the dummy column and the original
        assert all(
            [
                s1 == s2 if (not np.isnan(s1) and not np.isnan(s2)) else True
                for s1, s2 in zip(paraphrase_detect, generation_detect)
            ]
        )
        dataset = dataset.remove_columns(["w_wm_output_attacked", "w_wm_output_attacked_length"])
    return dataset


from utils.submitit import str2bool


def scheme_hparam_extractor(x):
    is_ff = "ff" in x
    is_simple_1 = ("simple_1" in x) or ("lefthash" in x)
    is_algorithm_3 = ("algorithm-3" in x) or ("selfhash" in x)
    is_anchored = "anchored" in x

    x = x.replace("ff-", "")
    x = x.replace("_prf", "")
    x = x.replace("anchored_", "")

    tup_x = x.split("-")

    # turn into a dict repr

    if is_ff:
        x_dict = {
            "prf_type": tup_x[0],
            "anchored": is_anchored,
            "context_width": int(tup_x[1]),
            "self_salt": str2bool(tup_x[2]),
        }
    elif is_simple_1:
        x_dict = {
            "prf_type": "additive",
            "anchored": False,
            "context_width": 1,
            "self_salt": False,
        }
    elif is_algorithm_3:
        x_dict = {
            "prf_type": "minhash",
            "anchored": True,
            "context_width": 4,
            "self_salt": True,
        }
    else:
        raise ValueError(f"Invalid scheme name {x} found.")

    return x_dict
