from datasets import Dataset, IterableDataset
from utils.io import read_jsonlines
import os

prompts = {
    0: "",
    1: "Answer the following question in 200-300 words. Explain it like I'm five.\n\n",
}


def load_lfqa(args=None, path="./utils/data/lfqa.jsonl"):
    cols_to_load = ["prefix", "gold_completion", "title", "selftext", "q_id"]

    args.dataset_config_name = None
    args.dataset_split = None
    args.columns_to_remove = list(set(args.columns_to_remove + cols_to_load))

    def lfqa_generator():
        for ex in read_jsonlines(path):
            row = {k: ex[k] for k in cols_to_load}
            row["prefix"] = f"{prompts[args.prompt_id]}{row['prefix']}"
            yield row

    dataset = IterableDataset.from_generator(lfqa_generator)
    return dataset
