# 💧 [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226) 🔍

### [Demo](https://huggingface.co/spaces/tomg-group-umd/lm-watermarking) | [Paper](https://arxiv.org/abs/2301.10226)

Official implementation of the watermarking and detection algorithms presented in the paper:

"A Watermark for Large language Models" by _John Kirchenbauer*, Jonas Geiping*, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein_  

Implementation is based on the "logit processor" abstraction provided by the [huggingface/transformers 🤗](https://github.com/huggingface/transformers) library.

The `WatermarkLogitsProcessor` is designed to be readily compatible with any model that supports the `generate` API.
Any model that can be constructed using the `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM` factories _should_ be compatible.

### Repo contents

The core implementation is defined by the `WatermarkBase`, `WatermarkLogitsProcessor`, and `WatermarkDetector` classes in the file `watermark_processor.py`.
The `demo_watermark.py` script implements a gradio demo interface as well as minimum working example in the `main` function.

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

### Abstract Usage of the `WatermarkLogitsProcessor` and `WatermarkDetector`

Generate watermarked text:
```python
watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                               gamma=0.25,
                                               delta=2.0,
                                               seeding_scheme="simple_1")

tokenized_input = tokenizer(input_text).to(model.device)
# note that if the model is on cuda, then the input is on cuda
# and thus the watermarking rng is cuda-based.
# This is a different generator than the cpu-based rng in pytorch!

output_tokens = model.generate(**tokenized_input,
                               logits_processor=LogitsProcessorList([watermark_processor]))

# if decoder only model, then we need to isolate the
# newly generated tokens as only those are watermarked, the input/prompt is not
output_tokens = output_tokens[:,tokenized_input["input_ids"].shape[-1]:]

output_text = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
```

Detect watermarked text:
```python
watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=0.25, # should match original setting
                                        seeding_scheme="simple_1", # should match original setting
                                        device=model.device, # must match the original rng device type
                                        tokenizer=tokenizer,
                                        z_threshold=4.0,
                                        normalizers=[],
                                        ignore_repeated_bigrams=False)

score_dict = watermark_detector.detect(output_text) # or any other text of interest to analyze
```

### Pending Items

- More advanced hashing/seeding schemes beyond `simple_1` as detailed in paper
- Attack experiment code


Suggestions and PR's welcome 🙂
