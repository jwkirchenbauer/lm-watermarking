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

import openai
import random

from utils.dipper_attack_pipeline import generate_dipper_paraphrases

from utils.evaluation import OUTPUT_TEXT_COLUMN_NAMES
from utils.copy_paste_attack import single_insertion, triple_insertion_single_len, k_insertion_t_len

SUPPORTED_ATTACK_METHODS = ["gpt", "dipper", "copy-paste", "scramble"]


def scramble_attack(example, tokenizer=None, args=None):
    # check if the example is long enough to attack
    for column in ["w_wm_output", "no_wm_output"]:
        if not check_output_column_lengths(example, min_len=args.cp_attack_min_len):
            # # if not, copy the orig w_wm_output to w_wm_output_attacked
            # NOTE changing this to return "" so that those fail/we can filter out these examples
            example[f"{column}_attacked"] = ""
            example[f"{column}_attacked_length"] = 0
        else:
            sentences = example[column].split(".")
            random.shuffle(sentences)
            example[f"{column}_attacked"] = ".".join(sentences)
            example[f"{column}_attacked_length"] = len(
                tokenizer(example[f"{column}_attacked"])["input_ids"]
            )
    return example


def gpt_attack(example, attack_prompt=None, args=None):
    assert attack_prompt, "Prompt must be provided for GPT attack"

    gen_row = example

    if args.no_wm_attack:
        original_text = gen_row["no_wm_output"]
    else:
        original_text = gen_row["w_wm_output"]

    attacker_query = attack_prompt + original_text
    query_msg = {"role": "user", "content": attacker_query}

    from tenacity import retry, stop_after_attempt, wait_random_exponential

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(25))
    def completion_with_backoff(model, messages, temperature, max_tokens):
        return openai.ChatCompletion.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )

    outputs = completion_with_backoff(
        model=args.attack_model_name,
        messages=[query_msg],
        temperature=args.attack_temperature,
        max_tokens=args.attack_max_tokens,
    )

    attacked_text = outputs.choices[0].message.content
    assert (
        len(outputs.choices) == 1
    ), "OpenAI API returned more than one response, unexpected for length inference of the output"
    example["w_wm_output_attacked_length"] = outputs.usage.completion_tokens
    example["w_wm_output_attacked"] = attacked_text
    if args.verbose:
        print(f"\nOriginal text (T={example['w_wm_output_length']}):\n{original_text}")
        print(f"\nAttacked text (T={example['w_wm_output_attacked_length']}):\n{attacked_text}")

    return example


def dipper_attack(dataset, lex=None, order=None, args=None):
    dataset = generate_dipper_paraphrases(dataset, lex=lex, order=order, args=args)
    return dataset


def check_output_column_lengths(example, min_len=0):
    baseline_completion_len = example["baseline_completion_length"]
    no_wm_output_len = example["no_wm_output_length"]
    w_wm_output_len = example["w_wm_output_length"]
    conds = all(
        [
            baseline_completion_len >= min_len,
            no_wm_output_len >= min_len,
            w_wm_output_len >= min_len,
        ]
    )
    return conds


def tokenize_for_copy_paste(example, tokenizer=None, args=None):
    for text_col in OUTPUT_TEXT_COLUMN_NAMES:
        if text_col in example:
            example[f"{text_col}_tokd"] = tokenizer(
                example[text_col], return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
    return example


def copy_paste_attack(example, tokenizer=None, args=None):
    # check if the example is long enough to attack
    if not check_output_column_lengths(example, min_len=args.cp_attack_min_len):
        # # if not, copy the orig w_wm_output to w_wm_output_attacked
        # NOTE changing this to return "" so that those fail/we can filter out these examples
        example["w_wm_output_attacked"] = ""
        example["w_wm_output_attacked_length"] = 0
        return example

    # else, attack

    # Understanding the functionality:
    # we always write the result into the "w_wm_output_attacked" column
    # however depending on the detection method we're targeting, the
    # "src" and "dst" columns will be different. However,
    # the internal logic for these functions has old naming conventions of
    # watermarked always being the insertion src and no_watermark always being the dst

    tokenized_dst = example[f"{args.cp_attack_dst_col}_tokd"]
    tokenized_src = example[f"{args.cp_attack_src_col}_tokd"]
    min_token_count = min(len(tokenized_dst), len(tokenized_src))

    if args.cp_attack_type == "single-single":  # 1-t
        tokenized_attacked_output = single_insertion(
            args.cp_attack_insertion_len,
            min_token_count,
            tokenized_dst,
            tokenized_src,
        )
    elif args.cp_attack_type == "triple-single":  # 3-t
        tokenized_attacked_output = triple_insertion_single_len(
            args.cp_attack_insertion_len,
            min_token_count,
            tokenized_dst,
            tokenized_src,
        )
    elif args.cp_attack_type == "k-t":
        tokenized_attacked_output = k_insertion_t_len(
            args.cp_attack_num_insertions,  # k
            args.cp_attack_insertion_len,  # t
            min_token_count,
            tokenized_dst,
            tokenized_src,
            verbose=args.verbose,
        )
    elif args.cp_attack_type == "k-random":  # k-t | k>=3, t in [floor(T/2k), T/k)
        raise NotImplementedError(f"Attack type {args.cp_attack_type} not implemented")
    elif args.cp_attack_type == "triple-triple":  # 3-(k_1,k_2,k_3)
        raise NotImplementedError(f"Attack type {args.cp_attack_type} not implemented")
    else:
        raise ValueError(f"Invalid attack type: {args.cp_attack_type}")

    example["w_wm_output_attacked"] = tokenizer.batch_decode(
        [tokenized_attacked_output], skip_special_tokens=True
    )[0]
    example["w_wm_output_attacked_length"] = len(tokenized_attacked_output)

    return example
