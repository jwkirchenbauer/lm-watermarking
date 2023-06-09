# Basic imports
import os
import argparse
import re
import functools

from tqdm import tqdm
from statistics import mean

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
from sklearn.metrics import roc_curve, precision_recall_curve, auc

import cmasher as cmr

# ### Load the processed dataset/frame


import sys

sys.path.insert(0, "..")

from datasets import Dataset
from utils.io import read_jsonlines, load_jsonlines

import transformers

# some file i/o helpers
from utils.io import write_jsonlines, write_json

INPUT_DIR = "/cmlscratch/manlis/test/watermarking-root/input"
OUTPUT_DIR = "/cmlscratch/manlis/test/watermarking-root/output"


# 15 colorblind-friendly colors
COLORS = [
    "#0072B2",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#000000",
    "#0072B2",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
]


def tokenize_and_mask(
    text, span_length, pct, ceil_pct=False, buffer_size=1, mask_string="<<<mask>>>"
):
    if isinstance(text, str):
        tokens = text.split(" ")
    else:
        tokens = text
    mask_string = mask_string

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f"<extra_id_{num_filled}>"
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = " ".join(tokens)
    return text


def tokenize_and_mask_glm(
    text, span_length, pct, ceil_pct=False, buffer_size=1, mask_string="[MASK]"
):
    tokens = text.split(" ")
    mask_string = mask_string

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    text = " ".join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    outputs = mask_model.generate(
        **tokens,
        max_length=mask_tokenizer.model_max_length,
        do_sample=True,
        top_p=1.0,
        num_return_sequences=1,
        eos_token_id=stop_id,
    )
    # outputs = mask_model.generate(**tokens, max_length=mask_tokenizer.model_max_length, do_sample=True, top_p=1.0, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def replace_masks_glm(texts):
    # n_expected = [len([x for x in text.split() if x == '[MASK]']) for text in texts]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    tokens = mask_tokenizer.build_inputs_for_generation(
        tokens, max_gen_length=mask_tokenizer.model_max_length
    ).to("cuda")
    outputs = mask_model.generate(
        **tokens,
        max_length=mask_tokenizer.model_max_length,
        do_sample=True,
        top_p=1.0,
        num_return_sequences=1,
        eos_token_id=mask_tokenizer.eop_token_id,
    )
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(" ") for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(
    texts, span_length, pct, ceil_pct=False, mask_filling_model_name="t5-3b", do_chunk=False
):
    if "t5" in mask_filling_model_name:
        if do_chunk:
            texts = [x.split(" ") for x in texts]
            ## chunk long texts
            if max([len(x) for x in texts]) > 600:
                text_pieces = [
                    [t[: len(t) // 3] for t in texts],
                    [t[len(t) // 3 : 2 * len(t) // 3] for t in texts],
                    [t[2 * len(t) // 3 :] for t in texts],
                ]
            else:
                text_pieces = [[t[: len(t) // 2] for t in texts], [t[len(t) // 2 :] for t in texts]]

            perturbed_pieces = []
            for pieces in text_pieces:
                masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in pieces]
                raw_fills = replace_masks(masked_texts)
                extracted_fills = extract_fills(raw_fills)
                perturbed_pieces.append(apply_extracted_fills(masked_texts, extracted_fills))
            ## put the chunks together
            perturbed_texts = []
            for i in range(len(texts)):
                perturbed_texts.append(" ".join([p[i] for p in perturbed_pieces]))
        else:
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
            raw_fills = replace_masks(masked_texts)
            extracted_fills = extract_fills(raw_fills)
            perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
    # elif 'glm' in mask_filling_model_name:
    #     masked_texts = [tokenize_and_mask_glm(x, span_length, pct, ceil_pct) for x in texts]
    #     raw_fills = replace_masks_glm(masked_texts)
    #     extracted_fills = extract_fills(raw_fills)
    #     perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while "" in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == ""]
        print(f"WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].")
        if do_chunk:
            new_perturbed_pieces = []
            for pieces in text_pieces:
                masked_texts = [
                    tokenize_and_mask(x, span_length, pct, ceil_pct)
                    for idx, x in enumerate(pieces)
                    if idx in idxs
                ]
                raw_fills = replace_masks(masked_texts)
                extracted_fills = extract_fills(raw_fills)
                new_perturbed_pieces.append(apply_extracted_fills(masked_texts, extracted_fills))
            new_perturbed_texts = []
            for i in range(len(texts)):
                new_perturbed_texts.append(" ".join([p[i] for p in new_perturbed_pieces]))
        else:
            masked_texts = [
                tokenize_and_mask(x, span_length, pct, ceil_pct)
                for idx, x in enumerate(texts)
                if idx in idxs
            ]
            raw_fills = replace_masks(masked_texts)
            extracted_fills = extract_fills(raw_fills)
            new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1

    return perturbed_texts


def perturb_texts(
    texts, span_length, pct, mask_filling_model_name, ceil_pct=False, chunk_size=20, do_chunk=False
):
    chunk_size = chunk_size
    if "11b" in mask_filling_model_name:
        chunk_size //= 2

    outputs = []
    for i in tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(
            perturb_texts_(
                texts[i : i + chunk_size],
                span_length,
                pct,
                ceil_pct=ceil_pct,
                mask_filling_model_name=mask_filling_model_name,
                do_chunk=do_chunk,
            )
        )
    return outputs


# Get the log likelihood of each text under the base_model
def get_ll(text):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to("cuda")
        labels = tokenized.input_ids
        return -base_model(**tokenized, labels=labels).loss.item()


def get_lls(texts):
    return [get_ll(text) for text in texts]


def get_perturbation_results(
    span_length=10,
    chunk_size=50,
    n_perturbations=1,
    n_perturbation_rounds=1,
    pct_words_masked=0.3,
    data_split="wm",
    mask_filling_model_name="t5-3b",
    do_chunk=False,
    save_path="/cmlscratch/manlis/test/watermarking-root/output/detect-gpt",
):
    ## check if pre-computed results exist
    if os.path.isfile(os.path.join(save_path, f"perturbed_raw_texts_{n_perturbations}.jsonl")):
        results = load_jsonlines(
            os.path.join(save_path, f"perturbed_raw_texts_{n_perturbations}.jsonl")
        )
    else:
        base_model.cpu()
        mask_model.cuda()

        torch.manual_seed(0)
        np.random.seed(0)

        results = []
        original_text = df["baseline_completion"]
        if data_split == "wm":
            sampled_text = df["w_wm_output"]
        elif data_split == "no_wm":
            sampled_text = df["no_wm_output"]
        elif data_split == "no_wm_paraphrase":
            sampled_text = df["w_wm_output_attacked"]
        else:
            raise NotImplementedError(f"Unknown split: {data_split}")

        perturb_fn = functools.partial(
            perturb_texts,
            span_length=span_length,
            pct=pct_words_masked,
            mask_filling_model_name=mask_filling_model_name,
            chunk_size=chunk_size,
            do_chunk=do_chunk,
        )

        p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(n_perturbations)])
        p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])
        for _ in range(n_perturbation_rounds - 1):
            try:
                p_sampled_text, p_original_text = perturb_fn(p_sampled_text), perturb_fn(
                    p_original_text
                )
            except AssertionError:
                break

        assert (
            len(p_sampled_text) == len(sampled_text) * n_perturbations
        ), f"Expected {len(sampled_text) * n_perturbations} perturbed samples, got {len(p_sampled_text)}"
        assert (
            len(p_original_text) == len(original_text) * n_perturbations
        ), f"Expected {len(original_text) * n_perturbations} perturbed samples, got {len(p_original_text)}"

        for i, idx in enumerate(original_text.index):
            results.append(
                {
                    "original": original_text[idx],
                    "sampled": sampled_text[idx],
                    "perturbed_sampled": p_sampled_text[
                        i * n_perturbations : (i + 1) * n_perturbations
                    ],
                    "perturbed_original": p_original_text[
                        i * n_perturbations : (i + 1) * n_perturbations
                    ],
                }
            )

        ## save perturbed samples in case job got preempted
        write_jsonlines(
            results, os.path.join(save_path, f"perturbed_raw_texts_{n_perturbations}.jsonl")
        )

    mask_model.cpu()
    base_model.cuda()

    for res in tqdm(results, desc="Computing log likelihoods"):
        p_sampled_ll = get_lls(res["perturbed_sampled"])
        p_original_ll = get_lls(res["perturbed_original"])
        res["original_ll"] = get_ll(res["original"])
        res["sampled_ll"] = get_ll(res["sampled"])
        res["all_perturbed_sampled_ll"] = p_sampled_ll
        res["all_perturbed_original_ll"] = p_original_ll
        res["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
        res["perturbed_original_ll"] = np.mean(p_original_ll)
        res["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
        res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1

    return results


def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve(
        [0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds
    )
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve(
        [0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds
    )
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)


def run_perturbation_experiment(
    results, criterion, span_length=10, n_perturbations=1, pct_words_masked=0.3, n_samples=500
):
    # compute diffs with perturbed
    predictions = {"real": [], "samples": []}
    for res in results:
        if criterion == "d":
            predictions["real"].append(res["original_ll"] - res["perturbed_original_ll"])
            predictions["samples"].append(res["sampled_ll"] - res["perturbed_sampled_ll"])
        elif criterion == "z":
            if res["perturbed_original_ll_std"] == 0:
                res["perturbed_original_ll_std"] = 1
                print("WARNING: std of perturbed original is 0, setting to 1")
                print(
                    f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}"
                )
                print(f"Original text: {res['original']}")
            if res["perturbed_sampled_ll_std"] == 0:
                res["perturbed_sampled_ll_std"] = 1
                print("WARNING: std of perturbed sampled is 0, setting to 1")
                print(
                    f"Number of unique perturbed sampled texts: {len(set(res['perturbed_sampled']))}"
                )
                print(f"Sampled text: {res['sampled']}")
            predictions["real"].append(
                (res["original_ll"] - res["perturbed_original_ll"])
                / res["perturbed_original_ll_std"]
            )
            predictions["samples"].append(
                (res["sampled_ll"] - res["perturbed_sampled_ll"]) / res["perturbed_sampled_ll_std"]
            )

    fpr, tpr, roc_auc = get_roc_metrics(predictions["real"], predictions["samples"])
    p, r, pr_auc = get_precision_recall_metrics(predictions["real"], predictions["samples"])
    name = f"perturbation_{n_perturbations}_{criterion}"
    print(f"{name} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        "name": name,
        "predictions": predictions,
        "info": {
            "pct_words_masked": pct_words_masked,
            "span_length": span_length,
            "n_perturbations": n_perturbations,
            "n_samples": n_samples,
        },
        "raw_results": results,
        "metrics": {
            "roc_auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr,
        },
        "pr_metrics": {
            "pr_auc": pr_auc,
            "precision": p,
            "recall": r,
        },
        "loss": 1 - pr_auc,
    }


## DetectGPT Running: get perturnation results
import json


# save the ROC curve for each experiment, given a list of output dictionaries, one for each experiment, using colorblind-friendly colors
def save_roc_curves(experiments, save_folder, args):
    # first, clear plt
    plt.clf()

    for experiment, color in zip(experiments, COLORS):
        metrics = experiment["metrics"]
        plt.plot(
            metrics["fpr"],
            metrics["tpr"],
            label=f"{experiment['name']}, roc_auc={metrics['roc_auc']:.3f}",
            color=color,
        )
        # print roc_auc for this experiment
        print(f"{experiment['name']} roc_auc: {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves ({args.base_model_name} - {args.mask_filling_model_name})")
    plt.legend(loc="lower right", fontsize=6)
    plt.savefig(f"{save_folder}/roc_curves.png")


def save_roc_curves_w_ztest(experiments, zscore_sample, zscore_original, save_folder, args):
    # first, clear plt
    plt.clf()

    ## make ztest ROC curve
    positive_preds = np.array(zscore_sample)
    negative_preds = np.array(zscore_original)
    positive_labels = np.ones_like(positive_preds, dtype=int)
    negative_labels = np.zeros_like(negative_preds, dtype=int)

    all_preds = np.concatenate((positive_preds, negative_preds))
    all_labels = np.concatenate((positive_labels, negative_labels))

    tpr_z, fpr_z, _ = roc_curve(all_labels, all_preds)
    roc_auc_z = auc(tpr_z, fpr_z)
    plt.plot(tpr_z, fpr_z, label=f"z-score test, roc_auc={roc_auc_z:.3f}")
    print(f"ztest roc_auc: {roc_auc_z:.3f}")

    for experiment, color in zip(experiments, COLORS):
        metrics = experiment["metrics"]
        plt.plot(
            metrics["fpr"],
            metrics["tpr"],
            label=f"{experiment['name']}, roc_auc={metrics['roc_auc']:.3f}",
            color=color,
        )
        # print roc_auc for this experiment
        print(f"{experiment['name']} roc_auc: {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves ({args.base_model_name} - {args.mask_filling_model_name})")
    plt.legend(loc="lower right", fontsize=6)
    plt.savefig(f"{save_folder}/roc_curves_w_ztests.png")


# save the histogram of log likelihoods in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed
def save_ll_histograms(experiments, save_folder):
    # first, clear plt
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)
            plt.hist([r["sampled_ll"] for r in results], alpha=0.5, bins="auto", label="sampled")
            plt.hist(
                [r["perturbed_sampled_ll"] for r in results],
                alpha=0.5,
                bins="auto",
                label="perturbed sampled",
            )
            plt.xlabel("log likelihood")
            plt.ylabel("count")
            plt.legend(loc="upper right")
            plt.subplot(1, 2, 2)
            plt.hist([r["original_ll"] for r in results], alpha=0.5, bins="auto", label="original")
            plt.hist(
                [r["perturbed_original_ll"] for r in results],
                alpha=0.5,
                bins="auto",
                label="perturbed original",
            )
            plt.xlabel("log likelihood")
            plt.ylabel("count")
            plt.legend(loc="upper right")
            plt.savefig(f"{save_folder}/ll_histograms_{experiment['name']}.png")
        except:
            pass


# save the histograms of log likelihood ratios in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed
def save_llr_histograms(experiments, save_folder):
    # first, clear plt
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)

            # compute the log likelihood ratio for each result
            for r in results:
                r["sampled_llr"] = r["sampled_ll"] - r["perturbed_sampled_ll"]
                r["original_llr"] = r["original_ll"] - r["perturbed_original_ll"]

            plt.hist([r["sampled_llr"] for r in results], alpha=0.5, bins="auto", label="sampled")
            plt.hist([r["original_llr"] for r in results], alpha=0.5, bins="auto", label="original")
            plt.xlabel("log likelihood ratio")
            plt.ylabel("count")
            plt.legend(loc="upper right")
            plt.savefig(f"{save_folder}/llr_histograms_{experiment['name']}.png")
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run detect-gpt with watermarked and baseline generations"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="facebook/opt-1.3b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--token_len",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--data_path",
        type=str,
    )
    parser.add_argument("--data_split", type=str, default="wm")
    parser.add_argument(
        "--mask_filling_model_name",
        type=str,
        default="t5-3b",
    )
    parser.add_argument("--n_positions", type=int, default=512)
    parser.add_argument(
        "--pct_words_masked",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--do_chunk",
        action="store_true",
        default=False,
    )
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--mask_top_p", type=float, default=1.0)
    parser.add_argument("--n_perturbation_list", type=str, default="1,10,100")
    parser.add_argument("--n_perturbation_rounds", type=int, default=1)
    parser.add_argument("--span_length", type=int, default=2)
    parser.add_argument("--buffer_size", type=int, default=1)

    args = parser.parse_args()
    if args.token_len > 300:
        args.do_chunk = True
    ## load data
    list_of_dict = load_jsonlines(args.data_path)
    raw_data = Dataset.from_list(list_of_dict)
    df = raw_data.to_pandas()
    ## drop samples that are too short
    original_len = len(df)
    print(f"Origianl #samples: {original_len}")
    if args.filter == "length":
        df = df[
            (df["baseline_completion_length"] == args.token_len)
            & (df["no_wm_output_length"] == args.token_len)
            & (df["w_wm_output_length"] == args.token_len)
        ]
        print(f" after filtering token length: {len(df)}")
    if args.filter == "null":
        try:
            df = df[
                (df["w_wm_output_length"].notnull())
                & (df["w_wm_output_attacked_length"].notnull())
                & ~(df["w_wm_output_length"] == "")
                & ~(df["w_wm_output_attacked_length"] == 0)
            ]
            print(f" after filtering token length: {len(df)}")
        except:
            print(
                "failed to filter null entries, probably because the file does not contain column 'w_wm_output_attacked_length'. "
            )
    args.n_samples = len(df)

    ## load models
    int8_kwargs = {}
    half_kwargs = {}
    if (
        "glm" not in args.mask_filling_model_name
    ):  # GLM uses an OP that's not supported in BFloat16: "triu_tril_cuda_template" not implemented for 'BFloat16'
        half_kwargs = dict(torch_dtype=torch.bfloat16)
    else:
        half_kwargs = dict(torch_dtype=torch.float16)

    ## load the base model (for generation) and base tokenizer
    optional_tok_kwargs = {}
    if "facebook/opt-" in args.base_model_name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs["fast"] = False
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model_name, **half_kwargs
    )
    base_model.eval()

    ####### load base tokenizer ########
    if "llama" in args.base_model_name:
        from transformers import LlamaTokenizer

        base_tokenizer = LlamaTokenizer.from_pretrained(
            args.base_model_name, padding_side="left", **optional_tok_kwargs
        )
    else:
        base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.base_model_name, padding_side="left", **optional_tok_kwargs
        )
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    print(f"Loading mask filling model {args.mask_filling_model_name}...")
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        args.mask_filling_model_name,
        **int8_kwargs,
        **half_kwargs,
        trust_remote_code="glm" in args.mask_filling_model_name,
    )
    mask_model.eval()

    ## mask model max length
    try:
        if "glm" in args.mask_filling_model_name:
            n_positions = mask_model.config.max_sequence_length
        else:
            n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512

    # if n_positions < args.token_len:
    # raise ValueError(f"Mask model cannot handle input longer then {n_positions}. Input token length: {args.token_len}")
    # preproc_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=n_positions)
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.mask_filling_model_name,
        model_max_length=n_positions,
        trust_remote_code="glm" in args.mask_filling_model_name,
    )
    mask_model.cpu()

    # perturbing text ops
    # define regex to match all <extra_id_*> tokens, where * is an integer
    pattern = re.compile(r"<extra_id_\d+>")

    SAVE_FOLDER = f'{OUTPUT_DIR}/detect-gpt/{os.path.basename(os.path.dirname(args.data_path))}-{args.data_split}-mask{args.mask_filling_model_name}/maskpct{args.pct_words_masked}-{os.path.basename(args.data_path).split(".")[0]}-ns{args.n_samples}'
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    outputs = []
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    for n_perturbations in n_perturbation_list:
        perturbation_results = get_perturbation_results(
            args.span_length,
            args.chunk_size,
            n_perturbations,
            args.n_perturbation_rounds,
            args.pct_words_masked,
            args.data_split,
            args.mask_filling_model_name,
            args.do_chunk,
            save_path=SAVE_FOLDER,
        )
        for perturbation_mode in ["d", "z"]:
            output = run_perturbation_experiment(
                perturbation_results,
                perturbation_mode,
                span_length=args.span_length,
                n_perturbations=n_perturbations,
                pct_words_masked=args.pct_words_masked,
                n_samples=args.n_samples,
            )
            outputs.append(output)
            ## write columns to the input df
            df[
                f"baseline_completion_detectgpt_score_{n_perturbations}_{perturbation_mode}"
            ] = output["predictions"]["real"]
            df[f"no_wm_output_detectgpt_score_{n_perturbations}_{perturbation_mode}"] = output[
                "predictions"
            ]["samples"]
            with open(
                os.path.join(
                    SAVE_FOLDER, f"perturbation_{n_perturbations}_{perturbation_mode}_results.json"
                ),
                "w",
            ) as f:
                json.dump(output, f)

    ## save the updated input df
    with open(os.path.join(SAVE_FOLDER, os.path.basename(args.data_path)), "w") as f:
        print(df.to_json(orient="records", lines=True), file=f, flush=False, end="")
    ## save meta file
    gen_table_meta = args.__dict__
    write_json(gen_table_meta, os.path.join(SAVE_FOLDER, "gen_table_meta.json"), indent=4)

    ### plot curves and histograms

    save_roc_curves(outputs, SAVE_FOLDER, args)
    # save_roc_curves_w_ztest(outputs, df["w_wm_output_z_score"],
    #                         df["baseline_completion_z_score"], SAVE_FOLDER, args)
    save_ll_histograms(outputs, SAVE_FOLDER)
    save_llr_histograms(outputs, SAVE_FOLDER)
