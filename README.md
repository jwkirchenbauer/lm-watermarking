# 💧 [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226) 🔍

### [Demo](https://huggingface.co/spaces/tomg-group-umd/lm-watermarking) | [Paper](https://arxiv.org/abs/2301.10226)

Official implementation of the watermarking and detection algorithms presented in the papers:

"A Watermark for Large language Models" by _John Kirchenbauer*, Jonas Geiping*, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein_  

"On the Reliability of Watermarks for Large Language Models" by _John Kirchenbauer*, Jonas Geiping*, Yuxin Wen, Manli Shu, Khalid Saifullah, Kezhi Kong, Kasun Fernando, Aniruddha Saha, Micah Goldblum, Tom Goldstein_

### Updates:

- **(6/7/23)** We're thrilled to announce the release of ["On the Reliability of Watermarks for Large Language Models"](https://arxiv.org/abs/2306.04634) Our new preprint documents a deep dive into the robustness properties of more advanced watermarks.

- **(6/9/23)** Initial code release implementing the alternate watermark and detector variants in the new work. Files located in this subdirectory: [`watermark_reliability_release`](watermark_reliability_release).

- **(9/23/23)** Update to the docs with recommendations on parameter settings. Extended implementation (recommended) available in `extended_watermark_processor.py`.

- **(1/16/24)** ["On the Reliability of Watermarks for Large Language Models"](https://arxiv.org/abs/2306.04634) has been accepted for publication and will be presented at ICLR 2024 in Vienna, Austria!

---

Implementation is based on the "logit processor" abstraction provided by the [huggingface/transformers 🤗](https://github.com/huggingface/transformers) library.

The `WatermarkLogitsProcessor` is designed to be readily compatible with any model that supports the `generate` API.
Any model that can be constructed using the `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM` factories _should_ be compatible.

### Repo contents

The core implementation is defined by the `WatermarkBase`, `WatermarkLogitsProcessor`, and `WatermarkDetector` classes in the files `watermark_processor.py`, for a minimal implementation and `extended_watermark_processor.py` for the more full featured implementation (recommended).
The `demo_watermark.py` script implements a gradio demo interface as well as minimum working example in the `main` function using the minimal version.

Details about the parameters and the detection outputs are provided in the gradio app markdown blocks as well as the argparse definition.

The `homoglyphs.py` and `normalizers.py` modules implement algorithms used by the `WatermarkDetector`. `homoglyphs.py` (and its raw data in `homoglyph_data`) is an updated version of the homoglyph code from the deprecated package described here: https://github.com/life4/homoglyphs.
The `experiments` directory contains pipeline code that we used to run the original experiments in the paper. However this is stale/deprecated
in favor of the implementation in `watermark_processor.py`.

### Demo Usage

As a quickstart, the app can be launched with default args (or deployed to a [huggingface Space](https://huggingface.co/spaces)) using `app.py`
which is just a thin wrapper around the demo script.
```sh
python app.py
gradio app.py # for hot reloading
# or
python demo_watermark.py --model_name_or_path facebook/opt-6.7b
```


### How to Watermark - A short guide on watermark hyperparameters
What watermark hyperparameters are optimal for your task or for a comparison to new watermarks? We'll provide a brief overview about all important settings below, and best practices for future work. This guide represents our current understanding of optimal settings as of August 2023, and so is a bit more up to date than our ICML 2023 conference paper.

**TL;DR**: As a baseline generation setting, we suggest default values of `gamma=0.25` and `delta=2.0`. Reduce delta if text quality is negatively impacted. For the context width, h, we recommend a moderate value, i.e. h=4, and as a default PRF we recommend `selfhash`, but can use `minhash` if you want. Reduce h if more robustness against edits is required. Note however that the choice of PRF only matters if h>1. The recommended PRF and context width can be easily selected by instantiating the watermark processor and detector with `seeding_scheme="selfhash"` (a shorthand for `seeding_scheme="ff-anchored_minhash_prf-4-True-15485863"`, but do use a different base key if actually deploying). For detection, always run with `--ignore--repeated-ngrams=True`.

1) **Logit bias delta**: The magnitude of delta determines the strength of the watermark. A sufficiently large value of delta recovers a "hard" watermark that encodes 1 bit of information at every token, but this is not an advisable setting, as it strongly affects model quality. A moderate delta in the range of [0.5, 2.0] is appropriate for normal use cases, but the strength of delta is relative to the entropy of the output distribution. Models that are overconfident, such as instruction-tuned models, may benefit from choosing a larger delta value. With non-infinite delta values, the watermark strength is directly proportional to the (spike) entropy of the text and exp(delta) (see Theorem 4.2 in our paper).

2) **Context width h**: Context width is the length of the context which is taken into account when seeding the watermark at each location. The longer the context, the "more random" the red/green list partitions are, and the less detectable the watermark is. For private watermarks, this implies that the watermark is harder to discover via brute-force (with an exponential increase in hardness with increasing context width h).
In the limit of a very long context width, we approach the "undetectable" setting of https://eprint.iacr.org/2023/763. However, the longer the context width, the less "nuclear" the watermark is, and robustness to paraphrasing and other attacks decreases. In the limit of h=0, the watermark is independent of local context and, as such, it is minimally random, but maximally robust against edits (see https://arxiv.org/abs/2306.17439).

3) **Ignoring repeated ngrams**: The watermark is only pseudo-random based on the local context. Whenever local context repeats, this constitutes a violation of the assumption that the PRNG numbers used to seed the green/red partition operation are drawn iid. (See Sec.4. in our paper for details). For this reason, p-values for text with repeated n-grams (n-gram here meaning context + chosen token) will be misleading. As such, detection should be run with `--ignore-repeated-ngrams` set to `True`. An additional, detailed analysis of this effect can be found in http://arxiv.org/abs/2308.00113.

4) **Choice of pseudo-random-function** (PRF): This choice is only relevant if context width h>1 and determines the robustness of the hash of the context against edits. In our experiments we find "min"-hash PRFs to be the most performant in striking a balance between maximizing robustness and minimizing impact on text quality. In comparison to a PRF that depends on the entire context, this PRF only depends on a single, randomly chosen token from the context.

5) **Self-Hashing**: It is possible to extend the context width of the watermark onto the current token. This effectively extends the context width "for-free" by one. The only downside is that this approach requires hashing all possible next tokens, and applying the logit bias only to tokens where their inclusion in the context would produce a hash that includes this token on the green list. This is slow in the way we implement it, because we use cuda's pseudorandom number generator and a simple inner-loop implementation, but in principle has a negligible cost, compared to generating new tokens if engineered for deployment. A generalized algorithm for self-hashing can be found as Alg.1 in http://arxiv.org/abs/2306.04634.

6) **Gamma**: Gamma denotes the fraction of the vocabulary that will be in each green list. We find gamma=0.25 to be slightly more optimal empirically, but this is a minor effect and reasonable values of gamma between 0.25 and 0.75 will lead to reasonable watermark. A intuitive argument can be made for why this makes it easier to achieve a fraction of green tokens sufficiently higher than gamma to reject the null hypothesis, when you choose a lower gamma value.

7) **Base Key**: Our watermark is salted with a small base key of 15485863 (the millionth prime). If you deploy this watermark, we do not advise re-using this key.

### How to use the watermark in your own code.

Our implementation can be added into any huggingface generation pipeline as an additional `LogitProcessor`, only the classes `WatermarkLogitsProcessor` and `WatermarkDetector` from the `extended_watermark_processor.py` file are required.

Example snippet to generate watermarked text:
```python

from extended_watermark_processor import WatermarkLogitsProcessor

watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                               gamma=0.25,
                                               delta=2.0,
                                               seeding_scheme="selfhash") #equivalent to `ff-anchored_minhash_prf-4-True-15485863`
# Note:
# You can turn off self-hashing by setting the seeding scheme to `minhash`.

tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)
# note that if the model is on cuda, then the input is on cuda
# and thus the watermarking rng is cuda-based.
# This is a different generator than the cpu-based rng in pytorch!

output_tokens = model.generate(**tokenized_input,
                               logits_processor=LogitsProcessorList([watermark_processor]))

# if decoder only model, then we need to isolate the
# newly generated tokens as only those are watermarked, the input/prompt is not
output_tokens = output_tokens[:,tokenized_input["input_ids"].shape[-1]:]

output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
```

Example snippet to detect watermarked text:
```python

from extended_watermark_processor import WatermarkDetector

watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=0.25, # should match original setting
                                        seeding_scheme="selfhash", # should match original setting
                                        device=model.device, # must match the original rng device type
                                        tokenizer=tokenizer,
                                        z_threshold=4.0,
                                        normalizers=[],
                                        ignore_repeated_ngrams=True)

score_dict = watermark_detector.detect(output_text) # or any other text of interest to analyze
```

To recover the main settings of the experiments in the original work (for historical reasons), use the seeding scheme `simple_1` and set `ignore_repeated_ngrams=False` at detection time.


### Contributing
Suggestions and PR's welcome 🙂
