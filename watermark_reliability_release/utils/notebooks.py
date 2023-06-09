# misc stuff for shortening notebooks

import pandas as pd
import numpy as np


def infer_length_column(base_col_name, dataframe, args=None):
    # in order of preference
    # the count computed at detection time is ideal, denoted `_num_tokens_scored`
    # else for the outputs it's generation-time token ct
    # and for the baseline its the initial ct base on tokenization and slice
    # both now called `_length`

    if args.ignore_repeated_ngrams:
        # if we're ignoring repeated ngrams, then we need to use the length column
        # since the num_tokens_scored column will be wrong/short
        # though this isn't a perfect solution bc there can be retokenization differences
        col_suffixes = ["_length"]
    else:
        col_suffixes = ["_num_tokens_scored", "_length"]

    for suf in col_suffixes:
        length_column_name = f"{base_col_name}{suf}"
        if length_column_name in dataframe.columns:
            return length_column_name

    raise ValueError(
        f"Could not find length column for {base_col_name}. Note, `_num_tokens_generated` suffix is deprecated in favor of `_length`."
    )


def filter_text_col_length(
    df, text_col_name=None, count_suffix="_num_tokens_scored", upper_T=205, lower_T=195
):
    assert text_col_name is not None
    text_col_prefix = text_col_name
    text_col_name = text_col_prefix + count_suffix

    # length filtering
    orig_len = len(df)

    df = df[(df[text_col_name] >= lower_T)]
    df = df[(df[text_col_name] <= upper_T)]

    print(f"Dropped {orig_len-len(df)} rows filtering {text_col_prefix}, new len {len(df)}")

    return df


def mega_filter(df):
    # drop retok_problematic_rows
    retok_problematic_rows = df[
        (df["w_bl_whitelist_fraction"] != -1.0)
        & (df["w_bl_whitelist_fraction"] != 1.0)
        & (df["bl_type"] == "hard")
    ]
    print(
        f"Num rows that are hard-blacklisted, and measureable, but still have a non-100% WL fraction: {len(retok_problematic_rows)} out of {len(df[df['bl_type'] == 'hard'])}"
    )

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
    df = df[((df["use_sampling"] == True) & (df["num_beams"] == 1)) | (df["use_sampling"] == False)]

    print(f"Dropped {orig_len-len(df)} rows, new len {len(df)}")

    # correct sampling temp
    df.loc[df["use_sampling"] == False, "sampling_temp"] = df.loc[
        df["use_sampling"] == False, "sampling_temp"
    ].fillna(0.0)
    df.loc[df["use_sampling"] == True, "sampling_temp"] = df.loc[
        df["use_sampling"] == True, "sampling_temp"
    ].fillna(1.0)

    # set to inf for hard blacklist
    df.loc[df["bl_type"] == "hard", "bl_logit_bias"] = np.inf
    # df.loc[df["bl_type"]=="hard","bl_logit_bias"] = 10000 # crosscheck with whats hardcoded in the bl processor

    # rename some stuff
    df["delta"] = df["bl_logit_bias"].values
    df["gamma"] = 1 - df["bl_proportion"].values
    df["gamma"] = df["gamma"].round(3)

    df["no_bl_act_num_wl_tokens"] = np.round(
        df["no_bl_whitelist_fraction"].values * df["no_bl_num_tokens_generated"], 1
    )  # round to 1 for sanity
    df["w_bl_act_num_wl_tokens"] = np.round(
        df["w_bl_whitelist_fraction"].values * df["w_bl_num_tokens_generated"], 1
    )  # round to 1 for sanity

    df["w_bl_std_num_wl_tokens"] = np.sqrt(df["w_bl_var_num_wl_tokens"].values)

    if "real_completion_length":
        df["baseline_num_tokens_generated"] = df["real_completion_length"].values

    if "actual_attacked_ratio" in df.columns:
        df["actual_attacked_fraction"] = (
            df["actual_attacked_ratio"].values * df["replace_ratio"].values
        )

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

    upper_T = 205
    lower_T = 195
    df = df[
        (df["baseline_hit_list_length"] >= lower_T)
        & (df["no_bl_hit_list_length"] >= lower_T)
        & (df["w_bl_hit_list_length"] >= lower_T)
    ]  # now also applies to the truncated version
    df = df[
        (df["baseline_hit_list_length"] <= upper_T)
        & (df["no_bl_hit_list_length"] <= upper_T)
        & (df["w_bl_hit_list_length"] <= upper_T)
    ]  # now also applies to the truncated version

    print(f"Dropped {orig_len-len(df)} rows, new len {len(df)}")

    return df
