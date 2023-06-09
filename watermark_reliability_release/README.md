# 💧2.0: [On the Reliability of Watermarks for Large Language Models](https://arxiv.org/abs/2306.04634)

This directory contains the codebase for reproducing the experiments in our [new 6/7/23 preprint](https://arxiv.org/abs/2306.04634).

### **NOTE**: this is a preliminary release, so please expect some small changes in the future as required.

---

The watermarking and watermark detection code itself is an extension of the `WatermarkLogitsProcessor` and `WatermarkDetector` classes released as part of the original work and contained in the root of the repository. Additional logic implementing a wider array of seeding schemes and alternate detection strategies is included and depended upon by the extended versions of the classes in this directory. 

To facilitate the broader array of experiments required for this study, an extra pipeline abstraction was implemented to manage the "generation", paraphrase "attack", and "evaluation" or detection phases. The general setup is that data, i.e. sets of generated samples, is written and read by each stage as "json lines" files `*.jsonl` with associated metadata files `*.json` to keep track of parameter settings used at each stage.

A prose version of usage instructions for the pipeline is described in a separate markdown file here: [PIPELINE.md](PIPELINE.md)

## wandb

The pipeline scripts, and in particular, the evaluation stage where detection is run and generation quality metrics are computed, are configured to push results to weights and biases (wandb). The figures in the paper are produced by:
1. sketching out the charts in wandb using filters and tags 
2. exporting/downloading the csv's of the data for each chart, and 
3. loading them in a notebook to format plots as necessary.

Alternately, the evaluation stage also saves a jsonl file where every line is a set of generations and all associated metrics and detection scores computed for it. This can also be loaded and analyzed manually in pandas, though the ROC space analyzes and average@T series for some metrics will have to be recomputed.

## llama

In order to use the llama model, you need to bring-your-own-weights, and then covert them to the huggingface format. 

