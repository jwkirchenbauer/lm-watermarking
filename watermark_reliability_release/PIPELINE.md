# Usage document for pipeline

6/7/23: Will be updated and built out as required.

## (1) **generate** a bunch of samples

The point of all this code is to construct pairwise examples 
of human text, unwatermarked, and watermarked text in something
resembling an unbiased or IID manner, despite the difficulty of this ask.

The key functionality is _oversampling_. A series of arguments control how
the raw datasets samples are turned into prompts, and then, provided
the raw prompts pass some checks, the prompts are
fed to the model, and the number of tokens naturally generated under normal
decoding, as well as watermark decoding. If the generations match the given 
(length) output filtering criteria, then the row "counts" as one of the `N`
requested samples.

Otherwise, the generations are stored, but the global counter of progress 
towards `N`, is not incremented, and thus this "overhead" is the cost
of being very restrictive in desiring "square" (`N` x `T`) shaped table of samples
in that all three of the human text, unwatermarked, and watermarked output columns
always have the same tokenized length.

At evaluation time, by default, all the point estimates, means, and ROC and AUC calculations are performed
on the subset of rows that all have about the target length (i.e. a subset with shape ~ `N` x `T`).

The `generation_pipeline.py` call in `run_pipeline.sh` demonstrates the basic usage.

### Key arguments controlling the oversampling logic...

### 'Shape' Controls

- `max_new_tokens`: an upperbound, i.e. target length `T=200`
- `min_prompt_tokens` : prompt len lower bound such as 50
- `min_generations` : the number of 'good' samples we'd like, ie `N=500`

### Prompt construction strategy

- `input_truncation_strategy`

One in `["completion_length", "prompt_length"]`. If the former, slices the end
`max_new_tokens` off of the raw sample to create the 'prompt' with the leading prefix (which can have variable length), making the `max_new_tokens` removed, the `baseline_completion`, or gold output.
If the latter, selects the leading `min_prompt_tokens` off of the raw sample as the prompt, 
leaving the remaining tokens (variable length) the `baseline_completion`.

### Filtering/oversampling criteria

- `input_filtering_strategy`: Can be one of `["completion_length", "prompt_length", "prompt_and_completion_length"]`.
In each case, if the relevant field doesn't meet the minimum criteria given by
`max_new_tokens` or `min_prompt_tokens` respectively, then the raw sample is thrown 
away before ever even being fed to the model.

- `output_filtering_strategy`: Can be one in `["no_filter", "max_new_tokens"]`, if the former, then no output filtering
is performed after generations are sampled from the model. However, if `max_new_tokens`
then each both the unwatermarked and watermarked generations are checked to ensure that 
they are at least `max_new_tokens` long. 

This is a subtle way of trying to adaptively collect samples (online, from any dataset) such that eventually we end up with at least a subset that matches the squareness (`N` x `T`) criteria we desire, without _forcing_ this to happen on every sample
by turning off the EOS token which amounts to a potentially
pathological distribution shift in the unwatermarked and watermarked output distributions
which would potentially confound generality of results.

Other generation args descriptions are explained by their argparse defintions, but these in particular control the watermarking:
- `seeding_scheme`: the watermarking embedding scheme being used, such as `lefthash` (formerly `simple_1`) or `selfhash` (formerly `algorithm-3` in reference to previous paper)
- `gamma`: parameter controlling size of the green partition for watermarking
- `delta`: parameter controlling how much bias is added to the green token logits before sampling

---

## (2) Optionally, apply an **attack** transformation to weaken the watermark, or make detection harder (for non-watermarking methods as well).

We implement three types of attacks in this pipeline: `gpt`, `dipper`,  and `copy-paste`. 
The key parameters for each are as follows:

- `gpt`: 
    -  `attack_model_name`: the OpenAI model variant to use
    -  `attack_prompt_id` : the index of the prompt to use, see `utils/prompts.json`
    -  `no_wm_attack`: whether to attack the un-watermarked generation column (`no_wm_output`). 
                       Default is the watermarked generation (`w_wm_output`)

- `dipper`: 
    - `lex`: lexical diversity knob for the dipper model/method
    - `order`: order diversity knob for the paraphrase attack

- `copy-paste`: 
    - `cp_attack_type`: k-t means `k` insertions of length `t`
    - `cp_attack_num_insertions`: `k` spec'd as an integer
    - `cp_attack_insertion_len`: `t` but generally spec'd as a percent of the full starting sequence length (i.e `25%`)
    - `cp_attack_src_col` : the sequence we're taking the tokens "to be detected" from , i.e. "positive" examples for 
                            the detector of interest. for watermarking this is `w_wm_output`
    - `cp_attack_dst_col` : the sequence we treat as "negative" surrounding context for the detector of interest. for watermarking this is `no_wm_output`.

All parameters have an associated help string in their argparse definition.

The `attack_pipeline.py` call in `run_pipeline.sh` demonstrates the basic usage of the attack functionality.

---

## (3) Run **evaluation** and watermark detection

This batches the process of applying a combination of metric
functions to the dataset of generations (jsonl) and returns a
new dataset of generations (jsonl) just with extra columns for a bunch of metrics.

This is separated from the generation phase to allow a given set of 
expensive generations to be reanalyzed in differnet ways with differnet metric
flavors as necessary.

The key parameters controlling metrics:


Key parameters and usage notes for detection:
- `evaluation_metrics`: a comma sep list of metrics to evaluate, such as `p-sp,repetition,diversity,z-score,windowed-z-score`
- `window_settings`: if running windowed detection specs the comma sep'd windowing strategies (such as `20,40,max`)
- `retrieval_technique`: if running retrieval detection, whether to use the `sim` or `bm25` strategy

All (other) parameters have a help string in their argparse definition.

The `evaluation_pipeline.py` call in `run_pipeline.sh` demonstrates the basic usage.

### Argument union and precedence

First, all arguments used at generation time (metadata file) are loaded by the
evaluation pipeline. Then the commandline args that were passed to the eval pipeline
are added via an update, or "overwriting union" operator, where all new args for
evaluation only are added to the current metadata object, but those that were
also present at generation time are _**overwritten**_ by those included in the 
evaluation argparse. 

If they match, then this is standard behavior. Overwriting shared arguments 
is disabled via the `overwrite_args` flag by default, but can be allowed this way.

Additionally, the code writes the metrics file into the same directory as the
generations file if only `input_dir` is passed. However, for safety clarity and organization,
one can pass an output dir in which to write the new dataset with metrics, as well
as the evaluation metadata as demonstrated in the `run_pipeline.sh` example.

---

## (3.1) Retrieval and DetectGPT detection

### Creating **prefixes**:

**Retrieval** detection is implemented as a metric, i.e. it is run by the evaluation script. To perform retrieval detection on full examples, nothing extra is required. To run retrieval at T, you first must run `broadcast_token_prefixes.py` with the `save_per_prefix` argument as `False` and with a `prefix_stride` of choice, such as 50, with a clean generation or attacked generation directory (with `jsonl` and meta file inside) as input. This will create a version of the dataset (new `jsonl` file) that contains all of the original rows, duplicated and then sliced to each prefix length defined by iterating by `prefix_stride` in the sequence length dimension.

For ex, if you have a file with `N=500` rows of length about `T=200` each, then running this script with `prefix_stride=50` would create a new file with `N=2000` where the first `500` rows all have length `50`, the next `500` have length `100` etc. If a given row say length `119` is too short for prefix length `i`, say the 3rd slice size in this example, `150`, then in the third block, it would be marked as `None`. This is to avoid any prefix block expected to be totally comprising a certain prefix length from containing a bunch of sequnces that are shorter than expected which confounds the measurement.

Now for **DetectGPT** a separate script, `detectgpt/detectgpt_main.py`, must be run pointing at a clean generation or attacked generation `jsonl` file. Additionally, to run detectgpt @ T, similar prefixing logic must be used. However, it must be run with `save_per_prefix` as `True` this time, which then creates a set of new files, each containing all the rows of the input `jsonl` file but trucated to each prefix length as described above. Then each run of the detectgpt script produces a new `jsonl` file (of length `N=500` in the above example) with the detectgpt score column added. Then, the notebook `join_jsonl_prefix_files.ipynb` can be used to join all those separate jsonl files for each individual prefix into one full file (`N=2000`).

### Running **detection**
For Retrieval detection, all that is necessary is to run the evaluation script on the `jsonl` containing all the prefixes, and point estimates for the detection at each prefix length will be created by grouping by the prefix length column and reducing. Note, the retrieval method will load only the full sequences into the retrieval database (by loading only the longest sample for each original row, so just `500` sequences in our example), but will query, or perform detection using all of the different prefixes.

For DetectGPT, the evaluation script must also be run, but with the `evaluation_metrics=detectgpt` alone, and no other metrics. This is because most of the script is a no-op at this point as every row already contains a detectgpt score and they just need to be turned into ROC plots or AUC measurements. As with retrieval detection, these will be automatically grouped by prefix length and reduced.
