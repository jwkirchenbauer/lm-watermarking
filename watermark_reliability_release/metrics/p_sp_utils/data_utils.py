from datasets import load_from_disk
import pandas as pd
import numpy as np

def get_df(data_dir):
    raw_data = load_from_disk(data_dir)
    # which is a hf dataset, and then can be converted to a dataframe
    df = raw_data.to_pandas()
    
    # drop retok_problematic_rows
    retok_problematic_rows = df[(df['w_bl_whitelist_fraction'] != -1.0) & (df['w_bl_whitelist_fraction'] != 1.0) & (df['bl_type'] == 'hard')]
    print(f"Num rows that are hard-blacklisted, and measureable, but still have a non-100% WL fraction: {len(retok_problematic_rows)} out of {len(df[df['bl_type'] == 'hard'])}")

    # drop special rows marked as -1.0
    orig_len = len(df)

    # df['no_bl_whitelist_fraction'].mask(df['no_bl_whitelist_fraction'] == -1.0, pd.NA, inplace=True)
    # df['w_bl_whitelist_fraction'].mask(df['w_bl_whitelist_fraction'] == -1.0, pd.NA, inplace=True)

    df = df[df["no_bl_whitelist_fraction"] != -1.0]
    df = df[df["w_bl_whitelist_fraction"] != -1.0]

    print(f"Dropped {orig_len-len(df)} rows, new len {len(df)}")

    # drop too few tokesn rows

    orig_len = len(df)
    # df = df[df["no_bl_ppl"].isna()]
    # df = df[df["w_bl_ppl"].isna()]
    df = df[~(df["no_bl_ppl"].isna() | df["w_bl_ppl"].isna())]
    print(f"Dropped {orig_len-len(df)} rows, new len {len(df)}")

    # drop huge biases
    orig_len = len(df)

    df = df[df["bl_logit_bias"] <= 100.0]

    print(f"Dropped {orig_len-len(df)} rows, new len {len(df)}")

    orig_len = len(df)

    # df = df[df["bl_hparams"].apply(lambda tup: (tup[0] == False and tup[2] != 1) or (tup[0] == True and tup[2] == 1) or (tup[0] == False))]
    df = df[((df["use_sampling"]==True) & (df["num_beams"] == 1)) | (df["use_sampling"]==False)]

    print(f"Dropped {orig_len-len(df)} rows, new len {len(df)}")

    # correct sampling temp
    df.loc[df["use_sampling"]==False,"sampling_temp"] = df.loc[df["use_sampling"]==False,"sampling_temp"].fillna(0.0)
    df.loc[df["use_sampling"]==True,"sampling_temp"] = df.loc[df["use_sampling"]==True,"sampling_temp"].fillna(1.0)

    # set to inf for hard blacklist
    df.loc[df["bl_type"]=="hard","bl_logit_bias"] = np.inf
    # df.loc[df["bl_type"]=="hard","bl_logit_bias"] = 10000 # crosscheck with whats hardcoded in the bl processor

    # rename some stuff
    df["delta"] = df["bl_logit_bias"].values
    df["gamma"] = 1 - df["bl_proportion"].values
    df["gamma"] = df["gamma"].round(3)

    df["no_bl_act_num_wl_tokens"] = np.round(df["no_bl_whitelist_fraction"].values*df["no_bl_num_tokens_generated"],1) # round to 1 for sanity
    df["w_bl_act_num_wl_tokens"] = np.round(df["w_bl_whitelist_fraction"].values*df["w_bl_num_tokens_generated"],1) # round to 1 for sanity

    df["w_bl_std_num_wl_tokens"] = np.sqrt(df["w_bl_var_num_wl_tokens"].values)

    if "real_completion_length":
        df["baseline_num_tokens_generated"] = df["real_completion_length"].values

    if "actual_attacked_ratio" in df.columns:
        df["actual_attacked_fraction"] = df["actual_attacked_ratio"].values*df["replace_ratio"].values

    if "meta" in df.columns:
        df["pile_set_name"] = df["meta"].apply(lambda dict: dict["pile_set_name"])

    df["baseline_hit_list_length"] = df["baseline_hit_list"].apply(len)
    df["no_bl_hit_list_length"] = df["no_bl_hit_list"].apply(len)
    df["w_bl_hit_list_length"] = df["w_bl_hit_list"].apply(len)


    # for pile outlier filtering
    df["w_bl_space_count"] = df["w_bl_output"].apply(lambda string: string.count(" "))
    df["no_bl_space_count"] = df["no_bl_output"].apply(lambda string: string.count(" "))
    df["baseline_space_count"] = df["baseline_completion"].apply(lambda string: string.count(" "))

    df["w_bl_space_frac"] = df["w_bl_space_count"].values / df["w_bl_hit_list_length"]
    df["no_bl_space_frac"] = df["no_bl_space_count"].values / df["no_bl_hit_list_length"]
    df["baseline_space_frac"] = df["baseline_space_count"].values / df["baseline_hit_list_length"]


    # Final length filtering 

    orig_len = len(df)

    # # main filters
    # # df = df[(df["real_completion_length"] == 200) & (df["w_bl_num_tokens_generated"] == 200)]
    # df = df[(df["gamma"] == 0.1) | (df["gamma"] == 0.25) | (df["gamma"] == 0.5)]
    # df = df[(df["delta"] == 1.0) | (df["delta"] == 2.0) | (df["delta"] == 10.0)]
    # df = df[(df["use_sampling"] == True)]
    # df = df[(df["bl_type"] == "soft")]

    # df = df[(df["real_completion_length"] == 200) & (df["no_bl_num_tokens_generated"] == 200) & (df["w_bl_num_tokens_generated"] == 200)] # now also applies to the truncated version
    # df = df[(df["no_bl_num_tokens_generated"] >= 500) & (df["w_bl_num_tokens_generated"] >= 500)] # all gas noop

    # # # attack specific
    # df = df[(df["real_completion_length"] == 200) & (df["no_bl_num_tokens_generated"] == 200) & (df["w_bl_num_tokens_generated"] == 200)]
    # df = df[(df["replace_ratio"] <= 0.7)]

    # # NOTE pile only
    # df = df[df["w_bl_space_frac"] <= 0.9]
    # df = df[df["no_bl_space_frac"] <= 0.9]
    # df = df[df["pile_set_name"] != "Github"]

    upper_T = 205
    lower_T = 195
    df = df[(df["baseline_hit_list_length"] >= lower_T) & (df["no_bl_hit_list_length"] >= lower_T) & (df["w_bl_hit_list_length"] >= lower_T)] # now also applies to the truncated version
    df = df[(df["baseline_hit_list_length"] <= upper_T) & (df["no_bl_hit_list_length"] <= upper_T) & (df["w_bl_hit_list_length"] <= upper_T)] # now also applies to the truncated version

    df.reset_index(inplace=True)

    print(f"Dropped {orig_len-len(df)} rows, new len {len(df)}")

    return df