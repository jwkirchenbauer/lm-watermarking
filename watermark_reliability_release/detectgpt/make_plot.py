# Basic imports
import os
import argparse
import json
import re

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
from sklearn.metrics import roc_curve, precision_recall_curve, auc


import sys
sys.path.insert(0, "..")

from datasets import Dataset
from utils.io import read_jsonlines, load_jsonlines

import transformers

from detectgpt_main import save_roc_curves_w_ztest


INPUT_DIR = "/cmlscratch/manlis/test/watermarking-root/input"
OUTPUT_DIR = "/cmlscratch/manlis/test/watermarking-root/output"

# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
                "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
                "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]


if __name__=="__main__":
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
        "--data_name",
        type=str,
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
        default=500,
    )
    parser.add_argument(
        "--data_path",
        type=str,
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="wm"
    )
    parser.add_argument(
        "--mask_filling_model_name",
        type=str,
        default="t5-3b",
    )
    parser.add_argument('--n_positions', type=int, default=512)
    parser.add_argument(
        "--pct_words_masked",
        type=float,
        default=0.3,
    )
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--n_perturbation_list', type=str, default="1,10,100")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--buffer_size', type=int, default=1)
    
    args = parser.parse_args()

    ## load data
    list_of_dict = load_jsonlines(args.data_path)
    raw_data = Dataset.from_list(list_of_dict)
    df = raw_data.to_pandas()
    ## drop samples that are too short
    original_len = len(df)
    df = df[(df["baseline_completion_length"] == args.token_len) \
            & (df["no_wm_num_tokens_generated"] == args.token_len) \
            & (df["w_wm_num_tokens_generated"] == args.token_len) ]
    print(f"Origianl #samples: {original_len}, after filtering token length: {len(df)}")
    args.n_samples = len(df)

    # perturbing text ops
    # define regex to match all <extra_id_*> tokens, where * is an integer
    pattern = re.compile(r"<extra_id_\d+>")

    SAVE_FOLDER = f'{OUTPUT_DIR}/detect-gpt/{args.data_name}-{args.data_split}-mask{args.mask_filling_model_name}/maskpct{args.pct_words_masked}-ns{args.n_samples}'
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    outputs = []

    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    for n_perturbations in n_perturbation_list:
        for perturbation_mode in ['d', 'z']:
            with open(os.path.join(SAVE_FOLDER, f"perturbation_{n_perturbations}_{perturbation_mode}_results.json"), "r") as f:
                output = json.load(f)
                outputs.append(output)


    ### plot curves and histograms
    
    save_roc_curves_w_ztest(outputs, df["w_wm_output_z_score"], 
                            df["baseline_completion_z_score"], SAVE_FOLDER, args, 
                            )