from datasets import load_dataset, IterableDataset


def load_wikitext(args=None):
    assert args is not None, "args must be provided to load_wikitext"
    assert (
        args.dataset_config_name is not None
    ), "args.dataset_config_name must be None to load_wikitext"
    assert args.dataset_split is not None, "args.dataset_split must be None to load_wikitext"

    args.columns_to_remove = list(set(args.columns_to_remove + ["text"]))

    # load the regular dataset
    raw_dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        split=args.dataset_split,
        streaming=False,  # we're doing this conversion ourselves
    )

    def wikitext_generator():
        # the generator loop
        for ex in raw_dataset:
            yield ex

    dataset = IterableDataset.from_generator(wikitext_generator)
    return dataset
