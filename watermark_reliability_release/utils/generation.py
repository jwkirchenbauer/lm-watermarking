# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

# HF classes

from datasets import load_dataset, IterableDataset

from torch import Tensor
from tokenizers import Tokenizer

from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
)

from .data.lfqa import load_lfqa
from .data.essays import load_essays
from .data.wikitext import load_wikitext

MAX_GENERATIONS = int(10000)  # Hardcoded max length to avoid infinite loop


def load_model(args):
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any(
        [(model_type in args.model_name_or_path) for model_type in ["t5", "T0"]]
    )
    args.is_decoder_only_model = any(
        [(model_type in args.model_name_or_path) for model_type in ["gpt", "opt", "bloom", "llama"]]
    )
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, torch_dtype=torch.float16, device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.load_fp16:
            pass
        else:
            model = model.to(device)
    else:
        device = "cpu"
    model.eval()

    if args.is_decoder_only_model:
        padding_side = "left"
    else:
        raise NotImplementedError(
            "Need to check how to handle padding for seq2seq models when calling generate"
        )

    if "llama" in args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(
            args.model_name_or_path, padding_side=padding_side
        )
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, padding_side=padding_side
        )

    args.model_max_length = model.config.max_position_embeddings

    return model, tokenizer, device


def add_idx(example, idx):
    example.update({"idx": idx})
    return example


def load_hf_dataset(args):
    dataset_name, dataset_config_name = args.dataset_name, args.dataset_config_name

    if dataset_name == "lfqa":
        dataset = load_lfqa(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": False,
                "input_col_name": "prefix",
                "ref_output_col_name": "gold_completion",
            }
        )
        # other args set within the load_lfqa function
    elif dataset_name == "wikitext":
        dataset = load_wikitext(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": True,
                "input_col_name": "text",
                "ref_output_col_name": None,
            }
        )
        # other args set within the load_wikitext function
    elif dataset_name == "essays":
        dataset = load_essays(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": False,
                "input_col_name": "instructions",
                "ref_output_col_name": "essays",
            }
        )
    elif dataset_name == "cml_pile":
        subsets = [dataset_config_name]
        dataset = load_dataset(
            "./data/cml_pile.py",
            subsets=subsets,
            streaming=args.stream_dataset,
            split=None,
            ignore_verifications=True,
        )[args.dataset_split]
        args.__dict__.update(
            {
                "truncate_input_for_prompt": True,
                "input_col_name": "text",
                "ref_output_col_name": None,
            }
        )
    else:
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            split=args.dataset_split,
            streaming=args.stream_dataset,
        )
        if "c4" in dataset_name:
            args.__dict__.update(
                {
                    "truncate_input_for_prompt": True,
                    "input_col_name": "text",
                    "ref_output_col_name": None,
                }
            )
            args.columns_to_remove = list(
                set(args.columns_to_remove + ["text", "timestamp", "url"])
            )
        elif "pile" in dataset_name:
            args.__dict__.update(
                {
                    "truncate_input_for_prompt": True,
                    "input_col_name": "text",
                    "ref_output_col_name": None,
                }
            )
            args.columns_to_remove = list(set(args.columns_to_remove + ["text", "meta"]))
        else:
            raise NotImplementedError(
                f"Dataset {dataset_name} not yet supported. Please add specs to load_hf_dataset function."
            )

    # add index to each row of dataset
    indexed_dataset = dataset.map(add_idx, batched=False, with_indices=True)

    # shuffle the first shuffle_buffer_size rows of streaming dataset, or whole dataset if not streaming
    # and take/select only the first n rows of the dataset (which caps the total number of pipeline iters possible)
    if isinstance(indexed_dataset, IterableDataset):
        shuffled_dataset = (
            indexed_dataset.shuffle(seed=args.shuffle_seed, buffer_size=args.shuffle_buffer_size)
            if args.shuffle_dataset
            else indexed_dataset
        )
        limited_dataset = (
            shuffled_dataset.take(args.limit_indices)
            if args.limit_indices is not None
            else shuffled_dataset
        )
    else:
        shuffled_dataset = (
            indexed_dataset.shuffle(seed=args.shuffle_seed)
            if args.shuffle_dataset
            else indexed_dataset
        )
        limited_dataset = (
            shuffled_dataset.select(range(args.limit_indices))
            if args.limit_indices is not None
            else shuffled_dataset
        )

    if args.limit_indices is None:
        try:
            args.limit_indices = len(limited_dataset)
        except Exception as e:
            # can't infer length of dataset, probably because it's an IterableDataset
            pass
    return limited_dataset


def check_input_lengths(
    example,
    min_sample_len=0,
    min_prompt_len=0,
    min_completion_len=0,
    max_input_len=None,
    max_new_tokens=None,
):
    orig_sample_length = example["orig_sample_length"]
    prompt_length = example["prompt_length"]
    real_completion_length = example["baseline_completion_length"]

    if max_input_len is not None:
        assert (
            max_new_tokens is not None
        ), "need to specify max_new_tokens if max_input_length is specified"

    conds = all(
        [
            orig_sample_length >= min_sample_len,
            prompt_length >= min_prompt_len,
            real_completion_length >= min_completion_len,
            (
                ((prompt_length + max_new_tokens) <= max_input_len)
                if max_input_len is not None
                else True
            ),
        ]
    )
    return conds


def check_output_lengths(example, min_output_len=0):
    # FIXME, maybe should check baseline completion length too
    no_wm_output_len = example["no_wm_output_length"]
    w_wm_output_len = example["w_wm_output_length"]
    conds = all(
        [
            no_wm_output_len >= min_output_len,
            w_wm_output_len >= min_output_len,
        ]
    )
    return conds


def tokenize_and_truncate(
    example: dict,
    input_col_name: str = "text",
    completion_length: int = None,
    prompt_length: int = None,
    hf_model_name: str = None,
    tokenizer=None,
    truncate_left=False,
    model_max_length=None,
):
    """take hf dataset entry and preprocess it for completion by a model"""
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert input_col_name in example, f"expects {input_col_name} field to be present"
    # tokenize
    inputs_ids = tokenizer(example[input_col_name], return_tensors="pt")["input_ids"]
    example.update({"untruncated_inputs": inputs_ids})

    if truncate_left:
        # truncate left
        inputs_ids = inputs_ids[:, -model_max_length:]
        if example["untruncated_inputs"].shape != inputs_ids.shape:
            print(
                "Input too long for model! ",
                "Left truncating under assumption that this is the prompt+output ",
                "to be fed to the *oracle* model",
            )
        example.update({"untruncated_inputs": inputs_ids})

    if (completion_length is not None) and (prompt_length is None):
        # leave at least one token as prefix # FIXME I think plus 1 since 0 is start tok
        slice_length = min(inputs_ids.shape[1] - 1, completion_length)
    elif (prompt_length is not None) and (completion_length is None):
        desired_comp_len = (inputs_ids.shape[1] - 1) - prompt_length
        slice_length = desired_comp_len if desired_comp_len > 0 else 0
    else:
        raise ValueError(
            (
                f"Can only tokenize and truncate based on either the desired prompt length or desired completion length,",
                f" but got completion_length:{completion_length},prompt_length:{prompt_length}",
            )
        )

    # truncate
    inputs_ids = inputs_ids[:, : inputs_ids.shape[1] - slice_length]
    # logic depending on special tokens for the model
    if "t5" in hf_model_name or "T0" in hf_model_name:
        inputs_ids[0, -1] = 1
    # else: pass
    example.update({"input_ids": inputs_ids})
    return example


def tokenize_only(
    example: dict,
    input_col_name: str = "text",
    ref_output_col_name: str = None,
    tokenize_ref_output: bool = False,
    hf_model_name: str = None,
    tokenizer=None,
    model_max_length=None,
):
    """take hf dataset entry and preprocess it for completion by a model
    (but don't truncate) where the dataset optionally has a secondary column
    that is the reference output to be scored against"""

    """take hf dataset entry and preprocess it for completion by a model"""
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert input_col_name in example, f"expects {input_col_name} field to be present"
    if ref_output_col_name is not None:
        assert ref_output_col_name in example, f"expects {ref_output_col_name} field to be present"

    # tokenize input
    input_ids = tokenizer(
        example[input_col_name], return_tensors="pt", truncation=True, max_length=model_max_length
    )["input_ids"]

    example.update({"input_ids": input_ids})

    if tokenize_ref_output:
        # NOTE not sure this logic is useful/required
        if ref_output_col_name is not None:
            # tokenize ref output
            ref_output_ids = tokenizer(
                example[ref_output_col_name],
                return_tensors="pt",
                truncation=True,
                max_length=model_max_length,
            )["input_ids"]

        tokd_input_len, tokd_ref_output_length = input_ids.shape[1], ref_output_ids.shape[1]
        if tokd_input_len + tokd_ref_output_length > model_max_length:
            # truncate the ref output
            original_ref_output_len = tokd_ref_output_length
            ref_output_ids = ref_output_ids[:, : model_max_length - tokd_input_len]
            if original_ref_output_len != ref_output_ids.shape[1]:
                print(
                    "Right truncating output, input+ref output too long for model. "
                    "Note, since this is generation time truncating the reference doesn't affect anything really."
                )
        example.update({"ref_output_ids": ref_output_ids})

    # logic depending on special tokens for the model
    if "t5" in hf_model_name or "T0" in hf_model_name:
        raise NotImplementedError("T5 style model not yet supported")

    return example


def tokenize_for_generation(
    example: dict,
    max_new_tokens: int = None,
    min_prompt_tokens: int = None,
    hf_model_name: str = None,
    tokenizer: Tokenizer = None,
    args: dict = None,
):
    # preprocessing, generation & scoring
    assert isinstance(example, dict), "Expect no batch dimension currently!"

    if not args.truncate_input_for_prompt:
        tokenize_ref_output = True  # NOTE, note really sure how necessary this is
        # preprocess for model generation/completion
        example = tokenize_only(
            example,
            input_col_name=args.input_col_name,
            ref_output_col_name=args.ref_output_col_name,
            hf_model_name=hf_model_name,
            tokenizer=tokenizer,
            model_max_length=args.model_max_length,
            tokenize_ref_output=tokenize_ref_output,
        )
        # Parse the results of tokenization. Simple, since
        # the prompt and baseline completion are from the raw text
        re_decoded_input = example[args.input_col_name]
        decoded_baseline_completion = example[args.ref_output_col_name]
        prompt_len = example["input_ids"].shape[1]
        baseline_completion_len = example["ref_output_ids"].shape[1]
        full_sample_len = prompt_len + baseline_completion_len
        # for now, remove this here, since it's not used downstream
        example.pop("ref_output_ids")
    else:
        # preprocess for model generation/completion
        example = tokenize_and_truncate(
            example,
            completion_length=max_new_tokens,
            prompt_length=min_prompt_tokens,
            hf_model_name=hf_model_name,
            tokenizer=tokenizer,
        )
        # Logic to parse the results of tokenzation and splitting to
        # construct string versions of the prompt and baseline completion
        inputs = example["input_ids"]
        prompt_len = inputs.shape[1]
        # for isolating the "gold" baseline completion
        untruncated_inputs = example.pop("untruncated_inputs")
        full_sample_len = untruncated_inputs.shape[1]
        # decode the preprocessed input to store for audit
        re_decoded_input = tokenizer.batch_decode(inputs, skip_special_tokens=True)[0]
        # also decode the original suffix of the input for audit as the baseline
        baseline_completion_tokens = untruncated_inputs[:, inputs.shape[-1] :]
        decoded_baseline_completion = tokenizer.batch_decode(
            baseline_completion_tokens, skip_special_tokens=True
        )[0]
        baseline_completion_len = full_sample_len - prompt_len

    example.update(
        {
            "truncated_input": re_decoded_input,
            "baseline_completion": decoded_baseline_completion,
            "orig_sample_length": full_sample_len,
            "prompt_length": prompt_len,
            "baseline_completion_length": baseline_completion_len,
        }
    )
    return example


def collate_batch(input_ids: list, collator: DataCollatorWithPadding = None):
    """collate batch of input_ids into a padded batch of tensors"""
    assert (
        input_ids[0].shape[0] == 1 and input_ids[0].shape[1] > 0
    ), "expecting batch dimension of each tensor to be 1"
    # remove batch dimension for each tensor
    input_ids = [x.squeeze(0) for x in input_ids]
    return collator({"input_ids": input_ids})["input_ids"]


def generate(
    examples,
    data_collator=None,
    generate_without_watermark=None,
    generate_with_watermark=None,
    watermark_processor=None,
    tokenizer=None,
    device=None,
    args=None,
):
    input_ids = collate_batch(input_ids=examples["input_ids"], collator=data_collator).to(device)

    with torch.no_grad():
        if args.generation_seed is not None:
            torch.manual_seed(args.generation_seed)
        output_without_watermark = generate_without_watermark(input_ids=input_ids)

        if args.generation_seed is not None:
            torch.manual_seed(args.generation_seed)
        output_with_watermark = generate_with_watermark(input_ids=input_ids)

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:, input_ids.shape[-1] :]
        output_with_watermark = output_with_watermark[:, input_ids.shape[-1] :]

    decoded_output_without_watermark = tokenizer.batch_decode(
        output_without_watermark, skip_special_tokens=True
    )
    decoded_output_with_watermark = tokenizer.batch_decode(
        output_with_watermark, skip_special_tokens=True
    )
    examples.update(
        {
            "no_wm_output": decoded_output_without_watermark,
            "w_wm_output": decoded_output_with_watermark,
            "no_wm_output_length": (output_without_watermark != tokenizer.pad_token_id)
            .sum(dim=-1)
            .tolist(),
            "w_wm_output_length": (output_with_watermark != tokenizer.pad_token_id)
            .sum(dim=-1)
            .tolist(),
        }
    )

    if watermark_processor.spike_entropies is not None:
        examples["spike_entropies"] = watermark_processor._get_and_clear_stored_spike_ents()
        examples["spike_entropies"] = [
            ents[:num_toks]
            for ents, num_toks in zip(examples["spike_entropies"], examples["w_wm_output_length"])
        ]

    return examples
