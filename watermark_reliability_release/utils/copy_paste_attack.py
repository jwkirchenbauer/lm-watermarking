import torch


def single_insertion(
    attack_len,
    min_token_count,
    tokenized_no_wm_output,  # dst
    tokenized_w_wm_output,  # src
):
    top_insert_loc = min_token_count - attack_len
    rand_insert_locs = torch.randint(low=0, high=top_insert_loc, size=(2,))

    # tokenized_no_wm_output_cloned = torch.clone(tokenized_no_wm_output) # used to be tensor
    tokenized_no_wm_output_cloned = torch.tensor(tokenized_no_wm_output)
    tokenized_w_wm_output = torch.tensor(tokenized_w_wm_output)

    tokenized_no_wm_output_cloned[
        rand_insert_locs[0].item() : rand_insert_locs[0].item() + attack_len
    ] = tokenized_w_wm_output[rand_insert_locs[1].item() : rand_insert_locs[1].item() + attack_len]

    return tokenized_no_wm_output_cloned


def triple_insertion_single_len(
    attack_len,
    min_token_count,
    tokenized_no_wm_output,  # dst
    tokenized_w_wm_output,  # src
):
    tmp_attack_lens = (attack_len, attack_len, attack_len)

    while True:
        rand_insert_locs = torch.randint(low=0, high=min_token_count, size=(len(tmp_attack_lens),))
        _, indices = torch.sort(rand_insert_locs)

        if (
            rand_insert_locs[indices[0]] + attack_len <= rand_insert_locs[indices[1]]
            and rand_insert_locs[indices[1]] + attack_len <= rand_insert_locs[indices[2]]
            and rand_insert_locs[indices[2]] + attack_len <= min_token_count
        ):
            break

    # replace watermarked sections into unwatermarked ones
    # tokenized_no_wm_output_cloned = torch.clone(tokenized_no_wm_output) # used to be tensor
    tokenized_no_wm_output_cloned = torch.tensor(tokenized_no_wm_output)
    tokenized_w_wm_output = torch.tensor(tokenized_w_wm_output)

    for i in range(len(tmp_attack_lens)):
        start_idx = rand_insert_locs[indices[i]]
        end_idx = rand_insert_locs[indices[i]] + attack_len

        tokenized_no_wm_output_cloned[start_idx:end_idx] = tokenized_w_wm_output[start_idx:end_idx]

    return tokenized_no_wm_output_cloned


def k_insertion_t_len(
    num_insertions,
    insertion_len,
    min_token_count,
    tokenized_dst_output,  # dst
    tokenized_src_output,  # src
    verbose=False,
):
    insertion_lengths = [insertion_len] * num_insertions

    # these aren't save to rely on indiv, need to use the min of both
    # dst_length = len(tokenized_dst_output)
    # src_length = len(tokenized_src_output) # not needed, on account of considering only min_token_count
    # as the max allowed index

    while True:
        rand_insert_locs = torch.randint(
            low=0, high=min_token_count, size=(len(insertion_lengths),)
        )
        _, indices = torch.sort(rand_insert_locs)

        if verbose:
            print(
                f"indices: {[rand_insert_locs[indices[i]] for i in range(len(insertion_lengths))]}"
            )
            print(
                f"gaps: {[rand_insert_locs[indices[i + 1]] - rand_insert_locs[indices[i]] for i in range(len(insertion_lengths) - 1)] + [min_token_count - rand_insert_locs[indices[-1]]]}"
            )

        # check for overlap condition for all insertions
        overlap = False
        for i in range(len(insertion_lengths) - 1):
            if (
                rand_insert_locs[indices[i]] + insertion_lengths[indices[i]]
                > rand_insert_locs[indices[i + 1]]
            ):
                overlap = True
                break

        if (
            not overlap
            and rand_insert_locs[indices[-1]] + insertion_lengths[indices[-1]] < min_token_count
        ):
            break

    # replace watermarked sections into unwatermarked ones

    tokenized_dst_output_cloned = torch.tensor(tokenized_dst_output)
    tokenized_src_output = torch.tensor(tokenized_src_output)

    for i in range(len(insertion_lengths)):
        start_idx = rand_insert_locs[indices[i]]
        end_idx = rand_insert_locs[indices[i]] + insertion_lengths[indices[i]]

        tokenized_dst_output_cloned[start_idx:end_idx] = tokenized_src_output[start_idx:end_idx]

    return tokenized_dst_output_cloned


##################################################
# Currently unused ###############################
##################################################

# def triple_insertion_triple_len(
#     tokenizer,
#     attacked_dict,
#     attack_lens,
#     min_token_count,
#     tokenized_no_wm_output,
#     tokenized_w_wm_output,
# ):
#     while True:
#         rand_insert_locs = torch.randint(low=0, high=min_token_count, size=(len(attack_lens),))
#         _, indices = torch.sort(rand_insert_locs)

#         if (
#             rand_insert_locs[indices[0]] + attack_lens[indices[0]] <= rand_insert_locs[indices[1]]
#             and rand_insert_locs[indices[1]] + attack_lens[indices[1]]
#             <= rand_insert_locs[indices[2]]
#             and rand_insert_locs[indices[2]] + attack_lens[indices[2]] <= min_token_count
#         ):
#             break

#     # replace watermarked sections into unwatermarked ones
#     tokenized_no_wm_output_cloned = torch.clone(tokenized_no_wm_output)
#     for i in range(len(attack_lens)):
#         start_idx = rand_insert_locs[indices[i]]
#         end_idx = rand_insert_locs[indices[i]] + attack_lens[indices[i]]

#         tokenized_no_wm_output_cloned[start_idx:end_idx] = tokenized_w_wm_output[start_idx:end_idx]

#     no_wm_output_cloned = tokenizer.batch_decode(
#         [tokenized_no_wm_output_cloned], skip_special_tokens=True
#     )[0]
#     attacked_dict["_".join([str(i) for i in attack_lens])] = no_wm_output_cloned


# refactored into attack.py
# # 200
# def copy_paste_attack(args, tokenizer, input_datas, attack_lens):
#     assert len(attack_lens) == 3

#     print(
#         f"total usable:",
#         torch.sum(
#             torch.tensor(
#                 [
#                     input_data["baseline_completion_length"] == 200
#                     and input_data["no_wm_num_tokens_generated"] == 200
#                     and input_data["w_wm_num_tokens_generated"] == 200
#                     for input_data in input_datas
#                 ]
#             )
#         ).item(),
#     )

#     output_datas = []
#     for input_data in input_datas:
#         if (
#             input_data["baseline_completion_length"] == 200
#             and input_data["no_wm_num_tokens_generated"] == 200
#             and input_data["w_wm_num_tokens_generated"] == 200
#         ):
#             pass
#         else:
#             continue

#         no_wm_output = input_data["no_wm_output"]
#         w_wm_output = input_data["w_wm_output"]

#         tokenized_no_wm_output = tokenizer(
#             no_wm_output, return_tensors="pt", add_special_tokens=False
#         )["input_ids"][0]
#         tokenized_w_wm_output = tokenizer(
#             w_wm_output, return_tensors="pt", add_special_tokens=False
#         )["input_ids"][0]

#         min_token_count = min(len(tokenized_no_wm_output), len(tokenized_w_wm_output))
#         attacked_dict = {}

#         attacked_dict["single_insertion"] = {}
#         single_insertion(
#             tokenizer,
#             attacked_dict["single_insertion"],
#             attack_lens[0],
#             min_token_count,
#             tokenized_no_wm_output,
#             tokenized_w_wm_output,
#         )

#         attacked_dict["triple_insertion_single_len"] = {}
#         triple_insertion_single_len(
#             tokenizer,
#             attacked_dict["triple_insertion_single_len"],
#             attack_lens[1],
#             min_token_count,
#             tokenized_no_wm_output,
#             tokenized_w_wm_output,
#         )

#         attacked_dict["triple_insertion_triple_len"] = {}
#         triple_insertion_triple_len(
#             tokenizer,
#             attacked_dict["triple_insertion_triple_len"],
#             attack_lens[2],
#             min_token_count,
#             tokenized_no_wm_output,
#             tokenized_w_wm_output,
#         )

#         input_data["w_wm_output_attacked"] = attacked_dict
#         output_datas.append(input_data)

#     return output_datas


# # 1000
# def copy_paste_attack_long(args, tokenizer, input_datas, attack_lens):
#     assert len(attack_lens) == 3

#     print(
#         f"total usable:",
#         torch.sum(
#             torch.tensor(
#                 [input_data["baseline_completion_length"] == 1000 for input_data in input_datas]
#             )
#         ).item(),
#     )

#     output_datas = []
#     residual_datas = []
#     for input_data in input_datas:
#         if len(output_datas) >= 500:
#             residual_datas.append(input_data)
#             continue

#         no_wm_output = input_data["baseline_completion"]
#         w_wm_output = input_data["w_wm_output"]

#         tokenized_no_wm_output = tokenizer(
#             no_wm_output, return_tensors="pt", add_special_tokens=False
#         )["input_ids"][0]
#         tokenized_w_wm_output = tokenizer(
#             w_wm_output, return_tensors="pt", add_special_tokens=False
#         )["input_ids"][0]

#         min_token_count = min(len(tokenized_no_wm_output), len(tokenized_w_wm_output))

#         if (
#             input_data["baseline_completion_length"] == 1000
#             and input_data["w_wm_num_tokens_generated"] > 500
#             and min_token_count > 500
#         ):
#             pass
#         else:
#             residual_datas.append(input_data)
#             continue

#         attacked_dict = {}

#         print("min_token_count:", min_token_count)

#         attacked_dict["single_insertion"] = {}
#         single_insertion(
#             tokenizer,
#             attacked_dict["single_insertion"],
#             attack_lens[0],
#             min_token_count,
#             tokenized_no_wm_output,
#             tokenized_w_wm_output,
#         )

#         attacked_dict["triple_insertion_single_len"] = {}
#         triple_insertion_single_len(
#             tokenizer,
#             attacked_dict["triple_insertion_single_len"],
#             attack_lens[1],
#             min_token_count,
#             tokenized_no_wm_output,
#             tokenized_w_wm_output,
#         )

#         attacked_dict["triple_insertion_triple_len"] = {}
#         triple_insertion_triple_len(
#             tokenizer,
#             attacked_dict["triple_insertion_triple_len"],
#             attack_lens[2],
#             min_token_count,
#             tokenized_no_wm_output,
#             tokenized_w_wm_output,
#         )

#         input_data["w_wm_output_attacked"] = attacked_dict
#         output_datas.append(input_data)

#     for idx, residual_data in enumerate(residual_datas):
#         output_datas[idx]["baseline_completion"] = residual_data["baseline_completion"]

#     return output_datas
