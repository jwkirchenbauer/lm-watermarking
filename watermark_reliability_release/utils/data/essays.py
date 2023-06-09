# https://huggingface.co/datasets/ChristophSchuhmann/essays-with-instructions

from datasets import Dataset, IterableDataset
from datasets import load_dataset

prompts = {
    0: "",
}


def load_essays(args=None):
    cols_to_load = ["instructions", "essays"]
    cols_to_remove = ["titles", "urls", "__index_level_0__"]

    dataset = load_dataset(
        "ChristophSchuhmann/essays-with-instructions",
        streaming=True,
        split=args.dataset_split,
    )
    dataset = dataset.remove_columns(cols_to_remove)

    args.dataset_config_name = None
    args.dataset_split = None
    args.columns_to_remove = list(set(args.columns_to_remove + cols_to_load))

    return dataset
