# Misc for hypothesis testing code used in notebooks and evaluation

import numpy as np
import numpy.ma as ma
from scipy.stats import geom, chisquare
import scipy


def rle_T_and_F_runs(arr):
    """
    Return run lengths and the value repeated in the run, of a boolean array.
    This handles arrays with different values and counts up runs of each value.
    https://stackoverflow.com/a/69693227
    """
    n = len(arr)
    if n == 0:
        values = np.empty(0, dtype=arr.dtype)
        lengths = np.empty(0, dtype=np.int_)
    else:
        positions = np.concatenate([[-1], np.nonzero(arr[1:] != arr[:-1])[0], [n - 1]])
        lengths = positions[1:] - positions[:-1]
        values = arr[positions[1:]]

    return values, lengths


def rle_F_succ_T_runs(arr):
    """
    With the k=1,2,3 convention, where success is False, and failure is True,
    we want to count the number of flips required before a success, i.e False.
    This 'resets' every time we see a False.
    Note, this truncates the tail of the array, so if the trailing elements are True,
    then they are not counted as a run since there is no terminating False.

    Note, this means if sequence is all True, then we return an empty array,
    and if the sequence is all False, then we return an ones array of length n.
    """
    n = len(arr)
    if n == 0:
        lengths = np.empty(0, dtype=np.int_)
    else:
        false_positions = np.concatenate([[-1], np.nonzero(arr == False)[0]])
        lengths = false_positions[1:] - false_positions[:-1]

    return lengths


def rle_T_succ_F_runs(arr):
    """
    Opposite above
    """
    n = len(arr)
    if n == 0:
        lengths = np.empty(0, dtype=np.int_)
    else:
        true_positions = np.concatenate([[-1], np.nonzero(arr == True)[0]])
        lengths = true_positions[1:] - true_positions[:-1]

    return lengths


def chi_squared_T_and_F_test(
    bool_arr=None,
    succ_prob=None,
    bin_spec=None,
    verbose=False,
    invert_bools=False,
    return_bin_counts=False,
    mask_zeros=False,
    lambda_="pearson",
    return_dict=False,
):
    assert bool_arr is not None, "bool_arr must be provided"
    assert succ_prob is not None, "succ_prob must be provided"

    if verbose:
        print(f"likelihood of success=F (1-gamma), or T run length geom dist 'p' = {succ_prob}")

    if invert_bools:
        bool_arr = ~bool_arr

    values, lengths = rle_T_and_F_runs(bool_arr)
    if verbose:
        print(f"Raw run lengths and their values and types:\n{lengths}\n{values}")

    remove_false = False
    remove_true = False
    if len(lengths) == 1:
        # lengths = np.array([len(bool_arr)+1]) # this is a HACK
        if values[0] == True:
            remove_false = True
            uniq_T_lens, T_run_counts = lengths, np.array([1])
            uniq_F_lens, F_run_counts = np.array([0]), np.array([0])
        elif values[0] == False:
            remove_true = True
            uniq_T_lens, T_run_counts = np.array([0]), np.array([0])
            uniq_F_lens, F_run_counts = lengths, np.array([1])
        else:
            raise ValueError("Unexpected value in bool array")
    else:
        uniq_T_lens, T_run_counts = np.unique(lengths[values == True], return_counts=True)
        uniq_F_lens, F_run_counts = np.unique(lengths[values == False], return_counts=True)

    if verbose:
        print("Unique T run lengths: ", uniq_T_lens)
    if verbose:
        print(f"Total T runs: {sum(T_run_counts)}")
    if verbose:
        print("Unique F run lengths: ", uniq_F_lens)
    if verbose:
        print(f"Total F runs: {sum(F_run_counts)}")

    if bin_spec == "max":
        largest_T_bin = max(uniq_T_lens)
        largest_F_bin = max(uniq_F_lens)
    elif bin_spec == "max_plus_1":
        largest_T_bin = max(uniq_T_lens) + 1
        largest_F_bin = max(uniq_F_lens) + 1
    elif isinstance(bin_spec, int):
        largest_T_bin = max(bin_spec, max(uniq_T_lens))
        largest_F_bin = max(bin_spec, max(uniq_F_lens))
    else:
        raise ValueError("bin_spec must be 'max' or an integer")

    if not remove_true:
        T_bins = np.arange(1, largest_T_bin + 1)
        if verbose:
            print("T Length bins: ", T_bins)
        obs_T_counts = np.zeros_like(T_bins, dtype=float)
        obs_T_counts[uniq_T_lens - 1] = np.array(T_run_counts, dtype=float)
        total_T_runs = sum(obs_T_counts)
    else:
        T_bins = uniq_T_lens
        if verbose:
            print("Ignoring lack of T runs in combined arrays")
        obs_T_counts = np.array([])
        total_T_runs = 0
    if not remove_false:
        F_bins = np.arange(1, largest_F_bin + 1)
        if verbose:
            print("F Length bins: ", F_bins)
        obs_F_counts = np.zeros_like(F_bins, dtype=float)
        obs_F_counts[uniq_F_lens - 1] = np.array(F_run_counts, dtype=float)
        total_F_runs = sum(obs_F_counts)
    else:
        F_bins = uniq_F_lens
        if verbose:
            print("Ignoring lack of F runs in combined arrays")
        obs_F_counts = np.array([])
        total_F_runs = 0

    if bin_spec in ["max", "max_plus_1"]:
        T_densities = geom.pmf(T_bins, succ_prob)
        T_densities[-1] += geom.sf(T_bins[-1], succ_prob)
        exp_T_counts = T_densities * total_T_runs

        F_densities = geom.pmf(F_bins, 1 - succ_prob)
        F_densities[-1] += geom.sf(F_bins[-1], 1 - succ_prob)
        exp_F_counts = F_densities * total_F_runs
    else:
        T_densities = geom.pmf(T_bins, succ_prob)
        exp_T_counts = T_densities * total_T_runs

        F_densities = geom.pmf(F_bins, 1 - succ_prob)
        exp_F_counts = F_densities * total_F_runs

    if remove_true:
        exp_T_counts = np.array([])
    if remove_false:
        exp_F_counts = np.array([])

    if verbose:
        print("Obs T counts: ", obs_T_counts)
    if verbose:
        print("Exp T counts: ", exp_T_counts)
    if verbose:
        print(f"densities: sum={sum(T_densities)}, {T_densities}")
    if verbose:
        print("Obs F counts: ", obs_F_counts)
    if verbose:
        print("Exp F counts: ", exp_F_counts)
    if verbose:
        print(f"densities: sum={sum(F_densities)}, {F_densities}")

    # concat the T and F obs and exp arrays
    obs_counts = np.concatenate([obs_T_counts, obs_F_counts])
    exp_counts = np.concatenate([exp_T_counts, exp_F_counts])

    if mask_zeros:
        obs_counts = ma.masked_array(obs_counts, mask=(obs_counts == 0))

    if verbose:
        print("Joined Obs counts: ", obs_counts)
    if verbose:
        print("Joined Exp counts: ", exp_counts)

    if lambda_ == "g_test":
        statistic, p_val = scipy.stats.power_divergence(
            f_obs=obs_counts, f_exp=exp_counts, ddof=0, axis=0, lambda_=0
        )
    elif lambda_ == "cressie_read":
        statistic, p_val = scipy.stats.power_divergence(
            f_obs=obs_counts, f_exp=exp_counts, ddof=0, axis=0, lambda_=2 / 3
        )
    elif lambda_ == "pearson":
        statistic, p_val = chisquare(obs_counts, exp_counts)
    else:
        raise ValueError(f"unrecognized lambda_={lambda_}")

    statistic = float(statistic)
    p_val = float(p_val)

    if return_dict:
        if return_bin_counts:
            return {
                "statistic": statistic,
                "p_val": p_val,
                # "total_T_runs": total_T_runs,
                "T_bins": T_bins,
                "obs_T_counts": obs_T_counts,
                "exp_T_counts": exp_T_counts,
                # "total_F_runs": total_F_runs,
                "F_bins": F_bins,
                "obs_F_counts": obs_F_counts,
                "exp_F_counts": exp_F_counts,
            }
        return {
            "statistic": statistic,
            "p_val": p_val,
            # "total_T_runs": total_T_runs,
            # "total_F_runs": total_F_runs,
        }

    if return_bin_counts:
        return (
            statistic,
            p_val,
            total_T_runs,
            T_bins,
            obs_T_counts,
            exp_T_counts,
            total_F_runs,
            F_bins,
            obs_F_counts,
            exp_F_counts,
        )
    return statistic, p_val, total_T_runs + total_F_runs


T_and_F_runs_dummy_dict_w_bins = {
    "statistic": float("nan"),
    "p_val": float("nan"),
    # "total_T_runs": float("nan"),
    "T_bins": [],
    "obs_T_counts": [],
    "exp_T_counts": [],
    # "total_F_runs": float("nan"),
    "F_bins": [],
    "obs_F_counts": [],
    "exp_F_counts": [],
}
T_and_F_runs_dummy_dict_no_bins = {
    "statistic": float("nan"),
    "p_val": float("nan"),
    # "total_T_runs": float("nan"),
    # "total_F_runs": float("nan"),
}


def chi_squared_runs_test(
    bool_arr=None,
    succ_prob=None,
    variant="F_succ_T_runs",
    bin_spec=200,
    verbose=False,
    invert_bools=False,
    return_bin_counts=False,
    mask_zeros=False,
    mask_leading_bins=0,
    diy=False,
    lambda_="pearson",
    return_dict=False,
):
    """
    Returns the chi squared statistic and p-value for the given data.
    The data is an array of run lengths, and a probability of success p.
    The variant is the convention for the run lengths, i.e. if success is False or True.
    The convention is that we are counting the number of flips required before a success.
    bin_spec is the number of bins to use for the chi squared test, if == "max" then we use the max run length.
    """
    assert bool_arr is not None, "bool_arr must be provided"
    assert succ_prob is not None, "succ_prob must be provided"

    if verbose:
        print(f"Boolean array: {bool_arr}")

    if variant == "F_succ_T_runs":
        run_func = rle_F_succ_T_runs
    elif variant == "T_succ_F_runs":
        run_func = rle_T_succ_F_runs
    elif variant == "T_and_F_runs":
        return chi_squared_T_and_F_test(
            bool_arr,
            succ_prob,
            bin_spec=bin_spec,
            verbose=verbose,
            invert_bools=invert_bools,
            return_bin_counts=return_bin_counts,
            mask_zeros=mask_zeros,
            lambda_=lambda_,
            return_dict=return_dict,
        )
    else:
        raise ValueError(f"unrecognized variant name={variant}")

    if invert_bools:
        bool_arr = ~bool_arr

    lengths = run_func(bool_arr)
    if len(lengths) == 0:
        lengths = np.array([len(bool_arr) + 1])  # this is a HACK
    uniq_lens, run_counts = np.unique(lengths, return_counts=True)

    if verbose:
        print("Unique run lengths: ", lengths)

    if verbose:
        print(f"Total runs: {sum(run_counts)}")

    if bin_spec == "max":
        largest_bin = max(uniq_lens)
    elif bin_spec == "max_plus_1":
        largest_bin = max(uniq_lens) + 1
    elif isinstance(bin_spec, int):
        largest_bin = max(bin_spec, max(uniq_lens))
    else:
        raise ValueError("bin_spec must be 'max' or an integer")

    bins = np.arange(1, largest_bin + 1)
    if verbose:
        print("Length bins: ", bins)

    obs_counts = np.zeros_like(bins, dtype=float)
    obs_counts[uniq_lens - 1] = np.array(run_counts, dtype=float)
    # total_runs = sum(obs_counts)

    if mask_zeros:
        # exp_counts = ma.masked_array(exp_counts, mask=(obs_counts==0))
        obs_counts = ma.masked_array(obs_counts, mask=(obs_counts == 0))

    if mask_leading_bins > 0:
        ones_mask = np.zeros_like(obs_counts)
        ones_mask[:mask_leading_bins] = 1
        obs_counts = ma.masked_array(obs_counts, mask=ones_mask)

    total_runs = obs_counts.sum()
    assert (
        total_runs > 0
    ), "total_runs must be > 0, this could be because all obs bins ended up masked"

    if bin_spec in ["max", "max_plus_1"]:
        densities = geom.pmf(bins, succ_prob)
        densities[-1] += geom.sf(bins[-1], succ_prob)
        # sub_cumulative_mass = 1.0 - densities[:mask_leading_bins].sum()
        # densities = densities * sub_cumulative_mass
        exp_counts = densities * total_runs
    else:
        densities = geom.pmf(bins, succ_prob)
        # sub_cumulative_mass = 1.0 - densities[:mask_leading_bins].sum()
        # densities = densities * sub_cumulative_mass
        exp_counts = densities * total_runs

    if mask_leading_bins > 0:
        # fixup the exp counts
        exp_counts[-mask_leading_bins - 1] += exp_counts[-mask_leading_bins:].sum()  # add tail
        exp_counts = np.concatenate((np.zeros(mask_leading_bins), exp_counts[:-mask_leading_bins]))
        exp_counts = ma.masked_array(
            exp_counts, mask=(exp_counts == 0)
        )  # this time we need this too

    if verbose:
        print("Obs counts: ", obs_counts)
    if verbose:
        print("Exp counts: ", exp_counts)
    if verbose:
        print(f"densities: sum={sum(densities)}, {densities}")

    # from scipy.stats import power_divergence
    # statistic, p_val = chisquare(obs_counts, exp_counts)
    if diy:
        # print("Local chi squared test")
        statistic, p_val = power_divergence(
            obs_counts, f_exp=exp_counts, ddof=0, axis=0, lambda_="pearson"
        )
    else:
        # print("Scipy chi squared test")
        if lambda_ == "g_test":
            statistic, p_val = scipy.stats.power_divergence(
                f_obs=obs_counts, f_exp=exp_counts, ddof=0, axis=0, lambda_=0
            )
        elif lambda_ == "cressie_read":
            statistic, p_val = scipy.stats.power_divergence(
                f_obs=obs_counts, f_exp=exp_counts, ddof=0, axis=0, lambda_=2 / 3
            )
        elif lambda_ == "pearson":
            statistic, p_val = chisquare(obs_counts, exp_counts)
        else:
            raise ValueError(f"unrecognized lambda_={lambda_}")

    statistic = float(statistic)
    p_val = float(p_val)

    if return_dict:
        if return_bin_counts:
            return {
                "statistic": statistic,
                "p_val": p_val,
                # "total_runs": total_runs,
                "bins": bins,
                "obs_counts": obs_counts,
                "exp_counts": exp_counts,
            }
        return {
            "statistic": statistic,
            "p_val": p_val,
            # "total_runs": total_runs,
        }
    if return_bin_counts:
        return statistic, p_val, total_runs, bins, obs_counts, exp_counts
    return statistic, p_val, total_runs


F_succ_T_runs_dummy_dict_w_bins = {
    "statistic": float("nan"),
    "p_val": float("nan"),
    # "total_runs": float("nan"),
    "bins": [],
    "obs_counts": [],
    "exp_counts": [],
}
F_succ_T_runs_dummy_dict_no_bins = {
    "statistic": float("nan"),
    "p_val": float("nan"),
    # "total_runs": float("nan"),
}
