# Basic imports
import os
from functools import partial
from argparse import Namespace

import numpy as np

# HF classses
from transformers import AutoTokenizer

from datasets import Dataset, concatenate_datasets


# watermarking micro lib
from watermark import (BlacklistLogitsProcessor,
                       compute_bl_metrics)

# some file i/o helpers
from io_utils import read_jsonlines, read_json


from watermark import compute_bl_metrics, BlacklistLogitsProcessor


###########################################################################
# Compute E[wl] for each example
###########################################################################

def expected_whitelist(example,
                           idx,
                           exp_wl_coef: float == None,
                           drop_spike_entropies: bool = False):
    assert "spike_entropies" in example, "Need to construct bl processor with store_spike_ents=True to compute them in post"

    num_toks_gend = example["w_bl_num_tokens_generated"]
    avg_spike_ent = np.mean(example["spike_entropies"])

    example.update({"avg_spike_entropy":avg_spike_ent})
    if drop_spike_entropies: del example["spike_entropies"]

    exp_num_wl = (exp_wl_coef*num_toks_gend)*avg_spike_ent
    var_num_wl = num_toks_gend*exp_wl_coef*avg_spike_ent*(1-(exp_wl_coef*avg_spike_ent))

    example.update({"w_bl_exp_num_wl_tokens":exp_num_wl})
    example.update({"w_bl_var_num_wl_tokens":var_num_wl})

    example.update({"exp_wl_coef":exp_wl_coef})

    if num_toks_gend > 0:
        example.update({"w_bl_exp_whitelist_fraction":exp_num_wl/num_toks_gend,
                        "w_bl_var_whitelist_fraction":var_num_wl/num_toks_gend})
    else:
        example.update({"w_bl_exp_whitelist_fraction":-1,
                        "w_bl_var_whitelist_fraction":-1})
    return example


from typing import Callable

def add_metadata(ex, meta_table=None):
    ex.update(meta_table)
    return ex


def str_replace_bug_check(example,idx):
    baseline_before = example["baseline_completion"]
    example["baseline_completion"] = baseline_before.replace(example["truncated_input"][:-1],"")
    if example["baseline_completion"] != baseline_before:
        print("baseline input replacement bug occurred, skipping row!")
        return False
    else:
        return True


def load_all_datasets(run_names: list[str]=None,
                      base_run_dir: str=None, 
                      meta_name: str=None, 
                      gen_name: str=None,
                      apply_metric_func: bool=False,
                      convert_to_pandas: bool = False,
                      drop_buggy_rows: bool = False,
                      limit_output_tokens: int = 0,
                      save_ds: bool = True,
                      save_dir: str=None):

    print(f"Loading {len(run_names)} datasets from {base_run_dir}...")
    
    if not isinstance(gen_name, Callable):
        file_check = lambda name: os.path.exists(f"{base_run_dir}/{name}/{gen_name}")
        assert all([file_check(name) for name in run_names]), f"Make sure all the run dirs contain the required data files: {meta_name} and {gen_name}"

    all_datasets = []
    for i,run_name in enumerate(run_names):

        print(f"[{i}] Loading dataset")

        run_base_dir = f"{base_run_dir}/{run_name}"
        gen_table_meta_path = f"{run_base_dir}/{meta_name}"
        
        if isinstance(gen_name, Callable):
            gen_table_path = f"{run_base_dir}/{gen_name(run_name)}"
        else:
            gen_table_path = f"{run_base_dir}/{gen_name}"

        # load the raw files
        gen_table_meta = read_json(gen_table_meta_path)
        gen_table_lst = [ex for ex in read_jsonlines(gen_table_path)]
        gen_table_ds = Dataset.from_list(gen_table_lst)

        print(f"Original dataset length={len(gen_table_ds)}")

        # drop the rows where the string replace thing happens
        if drop_buggy_rows:
            gen_table_ds_filtered = gen_table_ds.filter(str_replace_bug_check,batched=False,with_indices=True)
        else:
            gen_table_ds_filtered = gen_table_ds

        # enrich all rows with the run metadata
        add_meta = partial(
            add_metadata,
            meta_table=gen_table_meta
        )
        gen_table_w_meta = gen_table_ds_filtered.map(add_meta, batched=False)
        
        # optionally, apply the metric function(s) - somewhat expensive
        # want to do this here rather than at end because you need each run's tokenizer
        # though tbh it would be odd if they're not the same, but you can check that at the end
        if apply_metric_func:

            tokenizer = AutoTokenizer.from_pretrained(gen_table_meta["model_name"])

            comp_bl_metrics = partial(
                compute_bl_metrics,
                tokenizer=tokenizer,
                hf_model_name=gen_table_meta["model_name"],
                initial_seed=gen_table_meta["initial_seed"],
                dynamic_seed=gen_table_meta["dynamic_seed"],
                bl_proportion=gen_table_meta["bl_proportion"],
                use_cuda=True, # this is obvi critical to match the pseudorandomness
                record_hits=True,
                limit_output_tokens=limit_output_tokens,
            )
            gen_table_w_bl_metrics = gen_table_w_meta.map(comp_bl_metrics, batched=False, with_indices=True)

            
            # Construct the blacklist processor so you can get the expectation coef
            all_token_ids = list(tokenizer.get_vocab().values())
            vocab_size = len(all_token_ids)
            args = Namespace()
            args.__dict__.update(gen_table_meta)

            bl_processor = BlacklistLogitsProcessor(bad_words_ids=None, 
                                                    store_bl_ids=False, 
                                                    store_spike_ents=True, 
                                                    eos_token_id=tokenizer.eos_token_id, 
                                                    vocab=all_token_ids, 
                                                    vocab_size=vocab_size, 
                                                    bl_proportion=args.bl_proportion,
                                                    bl_logit_bias=args.bl_logit_bias,
                                                    bl_type=args.bl_type, 
                                                    initial_seed= args.initial_seed, 
                                                    dynamic_seed=args.dynamic_seed)
            
            if "spike_entropies" in gen_table_w_bl_metrics.column_names:
                comp_exp_num_wl = partial(
                    expected_whitelist,
                    exp_wl_coef=bl_processor.expected_wl_coef,
                    drop_spike_entropies=False,
                    # drop_spike_entropies=True,
                )
                gen_table_w_spike_ents = gen_table_w_bl_metrics.map(comp_exp_num_wl, batched=False, with_indices=True)
                final_single_run_ds = gen_table_w_spike_ents
            else:
                final_single_run_ds = gen_table_w_bl_metrics
        else:
            final_single_run_ds = gen_table_w_meta
        
        all_datasets.append(final_single_run_ds)
    
    ds = concatenate_datasets(all_datasets)
    
    if save_ds:
        ds.save_to_disk(save_dir)
    
    if convert_to_pandas:
        df = ds.to_pandas()
        return df
    else:
        return ds


output_dir = "/cmlscratch/jkirchen/spiking-root/lm-blacklisting/output_large_sweep"
# output_dir = "/cmlscratch/jkirchen/spiking-root/lm-blacklisting/output_large_sweep_downsize"

# output_dir = "/cmlscratch/jkirchen/spiking-root/lm-blacklisting/output_large_sweep_downsize"
# output_dir = "/cmlscratch/jkirchen/spiking-root/lm-blacklisting/output_greedy_redo"
# output_dir = "/cmlscratch/jkirchen/spiking-root/lm-blacklisting/output_greedy_gamma_0-25"

run_names = list(filter(lambda name: os.path.exists(f"{output_dir}/{name}/gen_table_w_metrics.jsonl"), sorted(os.listdir(output_dir))))
run_names = list(filter(lambda name: "realnewslike" in name, run_names))
# run_names = list(filter(lambda name: "pile" in name, run_names))
# run_names = list(filter(lambda name: "c4_en" in name, run_names))


# output_dir = "/cmlscratch/jkirchen/spiking-root/lm-blacklisting/output_attacked_greedy_updated"
# # output_dir = "/cmlscratch/jkirchen/spiking-root/lm-blacklisting/output_attacked_new"
# run_names = list(filter(lambda name: os.path.exists(f"{output_dir}/{name}/gen_table_w{('_'+name) if 't5' in name else ''}_attack_metrics.jsonl"), sorted(os.listdir(output_dir))))
# run_names = list(filter(lambda name: os.path.exists(f"{output_dir}/{name}/gen_table_w_attack_metrics.jsonl"), sorted(os.listdir(output_dir))))

runs_to_load = run_names


print(len(run_names))
for name in run_names: print(name)

runs_ready = [os.path.exists(f"{output_dir}/{name}/gen_table_w_metrics.jsonl") for name in runs_to_load]
# runs_ready = [os.path.exists(f"{output_dir}/{name}/gen_table_w_attack_metrics.jsonl") for name in runs_to_load]
print(f"all runs ready? {all(runs_ready)}\n{runs_ready}")


# save_name = "analysis_ds_1-21_greedy_redo"
# save_name = "analysis_ds_1-21_greedy_redo_truncated"
# save_name = "analysis_ds_1-21_greedy_redo_truncated_sanity_check"
# save_name = "analysis_ds_1-19_realnews_1-3_v2_hitlist_check"
# save_name = "analysis_ds_1-20_more_attack"

# save_name = "analysis_ds_1-23_greedy_gamma_0-25_truncated"
# save_name = "analysis_ds_1-21_greedy_attacked_updated_truncated"

# save_name = "analysis_ds_1-23_pile_1-3"
# save_name = "analysis_ds_1-23_en_1-3"

save_name = "analysis_ds_1-30_realnews_2-7"

save_dir = f"input/{save_name}"

raw_data = load_all_datasets(run_names=runs_to_load,
                            base_run_dir=output_dir,
                            meta_name="gen_table_meta.json",
                            gen_name="gen_table_w_metrics.jsonl",
                            # gen_name="gen_table_w_attack_metrics.jsonl",
                            apply_metric_func=True,
                            # drop_buggy_rows=True,
                            drop_buggy_rows=False,
                            # limit_output_tokens=200,
                            convert_to_pandas=False,
                            save_ds=True,
                            save_dir=save_dir)

print(f"All finished with {save_dir}!!")
