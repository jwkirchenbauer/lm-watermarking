import argparse
import json
import shutil
import nltk
import numpy as np
import tqdm
from functools import partial
from retriv import SearchEngine
import pickle
import subprocess
import os

from transformers import AutoTokenizer
from metrics.detect_retrieval_utils.detection_utils import print_tpr_target, get_roc, f1_score
from metrics.detect_retrieval_utils.models import load_model
from metrics.detect_retrieval_utils.embed_sentences import embed_all
import torch

nltk.download("punkt")


def detect_retrieval(data, args=None):
    # class parser_args(dict):
    #     def __getattr__(self, key):
    #         return self[key]

    # args = parser_args(
    #     {
    #         # 'threshold': 0.75,
    #         # "total_tokens": 50, # jwk not using anymore
    #         # "technique": "bm25",  # jwk moving to pipeline
    #         # 'output_file': "lfqa-data/gpt2_xl_strength_0.0_frac_0.5_300_len_top_p_0.9.jsonl_pp",
    #         # 'base_model': "gpt2-xl",
    #         # 'sim_threshold': 0.75,
    #         # "target_fpr": 0.01,  # jwk not using anymore
    #         # 'paraphrase_type': 'lex_40_order_100',
    #     }
    # )

    # tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # load SIM model
    download_url = "http://www.cs.cmu.edu/~jwieting/paraphrase-at-scale-english.zip"
    download_dir = "./metrics/detect_retrieval_utils/"
    load_file = "./metrics/detect_retrieval_utils/model.para.lc.100.pt"
    # Check if the required files exist
    if not os.path.exists(load_file):
        # make a box around the print statement
        print("=====================================" * 2)
        print(
            "Pretrained model weights wasn't found, Downloading paraphrase-at-scale-english.zip..."
        )
        print("=====================================" * 2)
        # Download the zip file
        subprocess.run(["wget", download_url])

        # # Unzip the file
        subprocess.run(["unzip", "paraphrase-at-scale-english.zip", "-d", download_dir])

        # # Delete the zip file
        os.remove("paraphrase-at-scale-english.zip")

        # move all the files to the parent directory
        _ = [
            shutil.move(os.path.join(download_dir, "paraphrase-at-scale-english", f), download_dir)
            for f in os.listdir(os.path.join(download_dir, "paraphrase-at-scale-english"))
        ]

        # Delete the empty directory
        os.rmdir(os.path.join(download_dir, "paraphrase-at-scale-english"))

    sim_model = load_model(load_file)
    sim_model.eval()
    embedder = partial(embed_all, model=sim_model)

    gens_list = []
    cands = []
    truncate_tokens = 10000  # args.total_tokens

    gen_nones = []
    gold_nones = []
    pp0_nones = []

    db_col_dict = {}

    # iterate over data and tokenize each instance
    for ds_i, dd in tqdm.tqdm(enumerate(data), total=len(data)):
        if dd[args.retrieval_db_column] is None or dd[args.retrieval_db_column] == "":
            gen_tokens = ""
            gen_nones.append(ds_i)
        else:
            gen_tokens = dd[args.retrieval_db_column].split()

        # Note, the gold column is always the baseline_completion or human text column
        # and the paraphrase column is always the w_wm_output_attacked column
        # despite the fact that the retrieval_db_column could be
        # no_wm_output or w_wm_output
        if dd["baseline_completion"] is None or dd["baseline_completion"] == "":
            gold_tokens = ""
            gold_nones.append(ds_i)
        else:
            gold_tokens = dd["baseline_completion"].split()

        if dd["w_wm_output_attacked"] is None or dd["w_wm_output_attacked"] == "":
            pp0_tokens = ""
            pp0_nones.append(ds_i)
        else:
            pp0_tokens = dd["w_wm_output_attacked"].split()

        # min_len = min(len(gold_tokens), len(pp0_tokens), len(gen_tokens))
        non_empty_str_lens = [
            len(toks) for toks in [gold_tokens, pp0_tokens, gen_tokens] if toks != ""
        ]
        if len(non_empty_str_lens) == 0:
            min_len = 0
        else:
            min_len = min(non_empty_str_lens)
        gens_list.append(" ".join(gen_tokens[:min_len]))

        cands.append(
            {
                "generation": " ".join(gen_tokens[:min_len]),
                "human": " ".join(gold_tokens[:min_len]),
                "paraphrase": " ".join(pp0_tokens[:min_len]),
                "ds_i": ds_i,
            }
        )
        # note this is our 'idx' not the iteration ds_i
        if dd["idx"] not in db_col_dict:
            db_col_dict[dd["idx"]] = []
        db_col_dict[dd["idx"]].append(" ".join(gen_tokens[:min_len]))

    print("Number of gens: ", len(gens_list))
    print("Number of candidates: ", len(cands))

    # print number of not none gens as well
    print("Number of 'not None' gens: ", len(gens_list) - len(gen_nones))

    # filter the gens_list based on ds_i that are not None
    gen_nones_set = set(gen_nones)
    gens_list = [gens_list[i] for i in range(len(gens_list)) if i not in gen_nones_set]

    if not args.retrieval_db_load_all_prefixes:
        # now iterate over the db_col_dict and find the longest version of the db column
        # for each 'idx'
        for idx, db_col in db_col_dict.items():
            db_col_dict[idx] = max(db_col, key=len)
        # now take all the longest versions of the db column and store them in a list
        # as the new gens_list
        new_gens_list = list(db_col_dict.values())
        gens_list = new_gens_list

    # index the cand_gens
    if args.retrieval_technique == "sim":
        gen_vecs = embedder(sentences=gens_list, disable=True)
    elif args.retrieval_technique == "bm25":
        collection = [{"text": x, "id": f"doc_{i}"} for i, x in enumerate(gens_list)]
        se = SearchEngine()
        se.index(collection)

    # iterate over cands and get similarity scores
    human_detect = []
    paraphrase_detect = []
    generation_detect = []

    for cand_i, cand in tqdm.tqdm(enumerate(cands)):
        try:
            if args.retrieval_technique == "sim":
                cand_vecs = embedder(
                    sentences=[cand["human"], cand["paraphrase"], cand["generation"]], disable=True
                )
                # get similarity scores
                sim_matrix = np.matmul(cand_vecs, gen_vecs.T)
                norm_matrix = (
                    np.linalg.norm(cand_vecs, axis=1, keepdims=True)
                    * np.linalg.norm(gen_vecs, axis=1, keepdims=True).T
                )
                sim_scores = sim_matrix / norm_matrix

                max_sim_score = np.max(sim_scores, axis=1)
                human_detect.append(max_sim_score[0])
                paraphrase_detect.append(max_sim_score[1])
                generation_detect.append(max_sim_score[2])

            elif args.retrieval_technique == "bm25":
                try:
                    res1 = se.search(cand["human"])[0]
                except:
                    res1 = {"text": ""}
                try:
                    res2 = se.search(cand["paraphrase"])[0]
                except:
                    res2 = {"text": ""}
                try:
                    res3 = se.search(cand["generation"])[0]
                except:
                    res3 = {"text": ""}
                human_detect.append(f1_score(cand["human"], res1["text"])[2])
                paraphrase_detect.append(f1_score(cand["paraphrase"], res2["text"])[2])
                generation_detect.append(f1_score(cand["generation"], res3["text"])[2])

        except Exception as e:
            print(
                f'Skipping this triple at cand_i={cand_i}: {cand["human"], cand["paraphrase"], cand["generation"]}'
            )
            print(e)
            human_detect.append(float("nan"))
            paraphrase_detect.append(float("nan"))
            generation_detect.append(float("nan"))

    # for ds_i in the human_detect, paraphrase_detect, and generation_detect that were None
    # set them to nan
    for ds_i in gen_nones:
        generation_detect[ds_i] = float("nan")
    for ds_i in gold_nones:
        human_detect[ds_i] = float("nan")
    for ds_i in pp0_nones:
        paraphrase_detect[ds_i] = float("nan")

    return human_detect, paraphrase_detect, generation_detect
