# micro lib to implement the watermarking extensions to LM generation
# as well as utils for eval/validaiton

from typing import List, Optional, Callable

import time
import random
import math
import torch
import numpy as np

from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor, LogitsProcessorList, set_seed


def tokenize_and_truncate(example: dict,
                          completion_length: int = None,
                          prompt_length: int = None,
                          hf_model_name: str = None,
                          tokenizer = None,
                          model_max_seq_len: int = 4096):
    """take hf dataset entry and preprocess it for completion by a model"""
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert "text" in example, "expects 'text' field to be present"
    # tokenize
    inputs = tokenizer.encode(example["text"], return_tensors="pt", truncation=True, max_length=model_max_seq_len)
    example.update({"untruncated_inputs": inputs})

    if (completion_length is not None) and (prompt_length is None):
        # leave at least one token as prefix # FIXME I think plus 1 since 0 is start tok
        slice_length = min(inputs.shape[1]-1, completion_length)
    elif (prompt_length is not None) and (completion_length is None):
        desired_comp_len = (inputs.shape[1]-1) - prompt_length
        slice_length = desired_comp_len if desired_comp_len > 0 else 0
    else:
        raise ValueError((f"Can only tokenize and truncate based on either the desired prompt length or desired completion length,",
                          f" but got completion_length:{completion_length},prompt_length:{prompt_length}"))

    # truncate
    inputs = inputs[:,:inputs.shape[1]-slice_length]
    # logic depending on special tokens for the model
    if "t5" in hf_model_name or "T0" in hf_model_name: 
        inputs[0,-1] = 1
    # else: pass
    example.update({"inputs": inputs})
    return example


class BlacklistLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that enforces that specified sequences will never be sampled.

    Args:
        bad_words_ids (`List[List[int]]`):
            List of list of token ids that are not allowed to be generated. In order to get the token ids of the words
            that should not appear in the generated text, use `tokenizer(bad_words, add_prefix_space=True,
            add_special_tokens=False).input_ids`.
        eos_token_id (`int`):
            The id of the *end-of-sequence* token.
    """

    def __init__(self, 
                bad_words_ids: List[List[int]], 
                eos_token_id: int,
                vocab: list[int], 
                vocab_size: int,
                bl_proportion: float=0.5,
                bl_logit_bias: float=1.0,
                bl_type: str = "hard", # "soft"
                initial_seed: int=None, 
                dynamic_seed: str=None, # "initial", "markov_1", None
                store_bl_ids: bool=True,
                store_spike_ents: bool = False,
                noop_blacklist: bool = False,
                ):
        
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.bl_proportion = bl_proportion
        self.bl_logit_bias = bl_logit_bias
        self.bl_type = bl_type

        if initial_seed is None: 
            self.initial_seed = None
            assert dynamic_seed != "initial"
        else:
            random.seed(initial_seed)
            self.initial_seed = initial_seed
            
        self.dynamic_seed = dynamic_seed

        self.eos_token_id = eos_token_id
        # self.bad_words_id_length_1 = self._prepare_bad_words(bad_words_ids)

        self.bad_words_mask: Optional[torch.LongTensor] = None

        self.store_bl_ids = store_bl_ids
        self.bl_ids = None

        self.store_spike_ents = store_spike_ents
        self.spike_entropies = None

        # hack to replace this with an approximation of infinite bias
        # so that the expectation coefficient will come out to 1.0
        if self.bl_type == "hard":
            self.bl_logit_bias = 10000 # FIXME to a value that is actually close to the largest soft watermark used

        alpha = torch.exp(torch.tensor(self.bl_logit_bias)).item()
        # gamma = self.bl_proportion
        gamma = 1.0-self.bl_proportion
        self.alpha = alpha
        self.gamma = gamma

        self.z_value = ((1-gamma)*(alpha-1))/(1-gamma+(alpha*gamma))
        self.expected_wl_coef = (gamma*alpha)/(1-gamma+(alpha*gamma))

        # catch for overflow when bias is "infinite"
        if self.alpha == torch.inf:
            self.z_value = 1.0
            self.expected_wl_coef = 1.0

        self.noop_blacklist = noop_blacklist
        if self.noop_blacklist: print(f"Blacklist processor for accounting only, no rescoring of logits")

        self.g_cuda = None
        self.large_prime = 15485863

    @property
    def blacklisted_ids(self):
        assert self.store_bl_ids, "Need to instantiate processor with `store_bl_ids` to be able to retrieve them later"
        # flatten the each indexes blacklist
        l_of_bl_ids = [[] for _ in range(len(self.bl_ids))]
        for b_idx, batch in enumerate(self.bl_ids):
            for l_of_l, seed in batch:
                bl_ids = [l[0] for l in l_of_l] # this was the main line, maybe unnecessary now?
                l_of_bl_ids[b_idx].append((bl_ids,seed))
        return l_of_bl_ids

    def get_and_clear_stored_bl_ids(self):
        old_bl_ids = self.bl_ids
        self.bl_ids = None
        return old_bl_ids

    def get_spike_entropies(self):
        spike_ents = [[] for _ in range(len(self.spike_entropies))]
        for b_idx, ent_tensor_list in enumerate(self.spike_entropies):
            for ent_tensor in ent_tensor_list:
                spike_ents[b_idx].append(ent_tensor.item())
        return spike_ents

    def get_and_clear_stored_spike_ents(self):
        spike_ents = self.get_spike_entropies()
        self.spike_entropies = None
        return spike_ents

    def compute_spike_entropy(self, scores):
        # precomputed z value in init
        probs = scores.softmax(dim=-1)
        denoms = 1+(self.z_value*probs)
        renormed_probs = probs / denoms
        sum_renormed_probs = renormed_probs.sum()
        return sum_renormed_probs

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        self.bad_words_id_length_1 = [None for _ in range(input_ids.shape[0])]

        if self.g_cuda is None:
            self.g_cuda = torch.Generator(device=input_ids.device)

        for b_idx in range(input_ids.shape[0]):
            if self.dynamic_seed == "initial":
                self.g_cuda.manual_seed(self.large_prime*self.initial_seed)
            elif self.dynamic_seed == "markov_1":
                self.g_cuda.manual_seed(self.large_prime*input_ids[b_idx][-1].item())
            elif self.dynamic_seed is None:
                # let the rng evolve naturally - this is not a realistic setting
                pass

            bl_ct = int(self.vocab_size*self.bl_proportion)
            blacklist_ids = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.g_cuda)[:bl_ct] # ty Yuxin :]

            if self.store_bl_ids: 
                if self.bl_ids is None: self.bl_ids = [[] for _ in range(input_ids.shape[0])]
                self.bl_ids[b_idx].append((blacklist_ids,input_ids.tolist()[b_idx][-1]))

            if self.store_spike_ents: 
                if self.spike_entropies is None: self.spike_entropies = [[] for _ in range(input_ids.shape[0])]
                self.spike_entropies[b_idx].append(self.compute_spike_entropy(scores[b_idx]))
            
            # self.bad_words_id_length_1[b_idx] = self._prepare_bad_words(blacklist_ids)
            # this logic may not really be necessary for our usecase
            self.bad_words_id_length_1[b_idx] = blacklist_ids
        
        if not self.noop_blacklist:
            self.bad_words_mask = self._calc_curr_bad_word_mask(scores)
            scores = self._set_scores_to_inf_for_banned_tokens(scores)

        return scores

    def _prepare_bad_words(self, bad_words_ids: List[List[int]]) -> list[int]:
        bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [self.eos_token_id], bad_words_ids))
        return bad_words_ids
        # used to have more logic, not used now

    def _calc_curr_bad_word_mask(self, scores: torch.FloatTensor) -> torch.BoolTensor:
        bad_words_mask = torch.zeros_like(scores)
        for b_idx in range(len(self.bad_words_id_length_1)):
            bad_words_mask[b_idx][self.bad_words_id_length_1[b_idx]] = 1
        final_mask = bad_words_mask.bool()
        return final_mask

    def _set_scores_to_inf_for_banned_tokens(
        self, scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be a
        list of list of banned tokens to ban in the format [[batch index, vocabulary position],...

        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
            # NOTE^^ Omitted logic for dynamic mask based on multi-token ban words
        """
        if self.bl_type == "hard":
            scores = scores.masked_fill(self.bad_words_mask, -float("inf"))
        elif self.bl_type == "soft":
            whitelist_mask = torch.logical_not(self.bad_words_mask)
            blacklist_mask = self.bad_words_mask
            scores[whitelist_mask] = scores[whitelist_mask] + self.bl_logit_bias
            # scores[blacklist_mask] = scores[blacklist_mask] - self.bl_logit_bias  # additive only
        else:
            raise NotImplementedError(f"unrecognized bl type {self.bl_type}!")          
        return scores


def score_sequence(inputs: Tensor = None, 
                   outputs: Tensor = None, 
                   tokenizer: Tokenizer = None,
                #    logits_processor: LogitsProcessor = None,
                   initial_seed: int = None,
                   dynamic_seed: str = None,
                   bl_proportion: float = None,
                   use_cuda: bool = True,
                   record_hits: bool = False,
                   debug: bool = True,
                #    trim_tokens: int = 2,
                ):

    assert  (inputs is not None) and \
            (outputs is not None) and \
            (tokenizer is not None),"output tensor, tokenizer, and bl params req'd"
            # (logits_processor is not None), 

    vocabulary = list(tokenizer.get_vocab().values())
    vocab_size = len(vocabulary)

    model_generations = outputs.tolist()[0] # these are tensors unpack once for speed
    # toks_generated = model_generations[num_orig_input_tokens:]
    toks_generated = model_generations
    num_toks_generated = len(toks_generated)

    # num_toks_to_trim = trim_tokens*2
    # if (num_toks_generated-num_toks_to_trim > 0) == False: 
    #     return -1, -1
        
    # assert num_toks_generated > num_toks_to_trim, f"Need more than {num_toks_to_trim} toks total since we trim start and end a bit."

    # toks_generated = toks_generated[trim_tokens:-trim_tokens]

    if initial_seed is not None:
        random.seed(initial_seed)
    
    device = (torch.device("cuda") if use_cuda else torch.device("cpu"))
    g_cuda = torch.Generator(device=device)
    large_prime = 15485863

    bl_hits, hit_list = 0, []

    prev_token = inputs[0][-1].item()
    # prev_token = toks_generated[0] # haven't decided whether this edge effect matters

    # for idx,tok_gend in enumerate(toks_generated[1:]):
    for idx,tok_gend in enumerate(toks_generated):

        # prev_token = model_generations[num_orig_input_tokens+idx-1]
        
        if dynamic_seed == "initial":
            g_cuda.manual_seed(large_prime*initial_seed)
        elif dynamic_seed == "markov_1":
            g_cuda.manual_seed(large_prime*prev_token)
        elif dynamic_seed is None:
            # let the rng evolve naturally - this is not a realistic setting
            pass

        bl_ct = int(vocab_size*bl_proportion)
        posthoc_blacklist = torch.randperm(vocab_size, device=device, generator=g_cuda)[:bl_ct] # ty Yuxin :]

        tok_in_ph_bl = tok_gend in posthoc_blacklist
        if tok_in_ph_bl:
            bl_hits += 1
            hit_list.append(True)
        else:
            hit_list.append(False)

        if debug: 
            decoded_token = tokenizer.decode(tok_gend, skip_special_tokens=True)
            print(f"Token generated: '{decoded_token}' was in the blacklist {tok_in_ph_bl}")
        
        prev_token = tok_gend

    if debug: 
        print(f"wl hits / num tokens : {num_toks_generated-bl_hits}/{num_toks_generated} = {(num_toks_generated-bl_hits)/num_toks_generated:.02f}")
        print(f"bl hits / num tokens : {bl_hits}/{num_toks_generated} = {bl_hits/num_toks_generated:.02f}")

    if record_hits:
        return bl_hits, num_toks_generated, hit_list
    # bl_fraction = bl_hits/num_toks_generated
    return bl_hits, num_toks_generated


def tokenize_for_generation(example: dict, 
                            idx: int,
                            max_new_tokens: int=None,
                            min_prompt_tokens: int=None,
                            hf_model_name : str=None,
                            tokenizer: Tokenizer=None,
                            model: torch.nn.Module=None):

    # preprocessing, generation & scoring
    assert isinstance(example, dict), "Expect no batch dimension currently!"

    # preprocess for model generation/completion
    example = tokenize_and_truncate(example, 
                                    completion_length=max_new_tokens, 
                                    prompt_length=min_prompt_tokens,
                                    hf_model_name=hf_model_name,
                                    tokenizer=tokenizer,
                                    # model_max_seq_len=model.config.max_position_embeddings)
                                    model_max_seq_len=None)
    inputs = example["inputs"]
    # for calculating the baseline violation rate across the "gold" completion
    untruncated_inputs = example["untruncated_inputs"]

    # decode the preprocessed input to store for audit
    re_decoded_input = tokenizer.batch_decode(inputs, skip_special_tokens=True)[0]
    example.update({"truncated_input":re_decoded_input})

    # also decode the original suffix of the input for audit as the baseline
    decoded_untruncated_input = tokenizer.batch_decode(untruncated_inputs, skip_special_tokens=True)[0]
    example.update({"baseline_completion":decoded_untruncated_input.replace(re_decoded_input,"")})

    example.update({
        "orig_sample_length"            : untruncated_inputs.shape[1],
        "prompt_length"                 : inputs.shape[1],
        "real_completion_length"        : untruncated_inputs.shape[1] - inputs.shape[1],
    })
    return example


def generate_completions(example: dict, 
                        idx: int,
                        max_new_tokens: int=None,
                        hf_model_name : str=None,
                        tokenizer: Tokenizer=None,
                        model: torch.nn.Module=None,
                        no_bl_partial: Callable=None,
                        w_bl_partial: Callable=None,
                        # return_logits: bool=False,
                        bl_processor_list: LogitsProcessorList=None):

    # preprocessing, generation & scoring
    assert isinstance(example, dict), "Expect no batch dimension currently!"

    # # preprocess for model generation/completion
    # example = tokenize_and_truncate(example, 
    #                                 completion_length=max_new_tokens, 
    #                                 hf_model_name=hf_model_name,
    #                                 tokenizer=tokenizer,
    #                                 model_max_seq_len=model.config.max_position_embeddings)
    # inputs = example["inputs"]
    # # for calculating the baseline violation rate across the "gold" completion
    # untruncated_inputs = example["untruncated_inputs"]

    # # decode the preprocessed input to store for audit
    # re_decoded_input = tokenizer.batch_decode(inputs, skip_special_tokens=True)[0]
    # example.update({"truncated_input":re_decoded_input})

    # # also decode the original suffix of the input for audit as the baseline
    # decoded_untruncated_input = tokenizer.batch_decode(untruncated_inputs, skip_special_tokens=True)[0]
    # example.update({"baseline_completion":decoded_untruncated_input.replace(re_decoded_input,"")})

    inputs = example["inputs"]
    re_decoded_input = example["truncated_input"]

    # call the vanilla and watermarked generation function wrappers with the preprocessed inputs
    with torch.no_grad():

        samples_taken = 0
        max_retries = 10
        success = False
        while (success is False) and (samples_taken < max_retries):
            samples_taken += 1
            
            # set_seed(1234) # debugging the error when using sampling # leaving this off for now

            start_generation = time.time()
            outputs_no_bl = no_bl_partial(inputs.to(model.device))
            example["no_bl_gen_time"] = time.time() - start_generation

            # set_seed(1234) # debugging the error when using sampling

            start_generation = time.time()
            outputs_w_bl = w_bl_partial(inputs.to(model.device))
            example["w_bl_gen_time"] = time.time() - start_generation

            # if return_logits:
            #     output_no_bl_dict = outputs_no_bl
            #     logits_no_bl = output_no_bl_dict.scores
            #     outputs_no_bl = output_no_bl_dict.sequences
            #     example["logits_no_bl"] = logits_no_bl

            #     output_w_bl_dict = outputs_w_bl
            #     logits_w_bl = output_w_bl_dict.scores
            #     outputs_w_bl = output_w_bl_dict.sequences
            #     example["logits_w_bl"] = logits_w_bl


            if bl_processor_list:
                if bl_processor_list[0].bl_ids is not None:
                    example["bl_ids"] = bl_processor_list[0].get_and_clear_stored_bl_ids()
                if bl_processor_list[0].spike_entropies is not None:
                    example["spike_entropies"] = bl_processor_list[0].get_and_clear_stored_spike_ents()

            try: 
                # decode and store the new generations for auditing
                no_bl_decoded_output = tokenizer.batch_decode(outputs_no_bl, skip_special_tokens=True)[0]
                example.update({"no_bl_output":no_bl_decoded_output.replace(re_decoded_input,"")})

                w_bl_decoded_output = tokenizer.batch_decode(outputs_w_bl, skip_special_tokens=True)[0]
                example.update({"w_bl_output":w_bl_decoded_output.replace(re_decoded_input,"")})
                
                success = True

            except:
                # log what happened
                print(f"Error while trying to decode the outputs of the model...")
                if samples_taken == 1:
                    print(f"truncated_input: {inputs.tolist()}")
                print(f"Result of attempt {samples_taken}")
                print(f"shape outputs_no_bl: {outputs_no_bl.shape}")
                no_bl_toks = outputs_no_bl.tolist()[0]
                print(f"outputs_no_bl: {no_bl_toks}")
                print(f"outputs_no_bl min: {min(no_bl_toks)}")
                print(f"outputs_no_bl max: {max(no_bl_toks)}")

                print(f"shape outputs_w_bl: {outputs_w_bl.shape}")
                w_bl_toks = outputs_w_bl.tolist()[0]
                print(f"outputs_w_bl: {w_bl_toks}")
                print(f"outputs_w_bl min: {min(w_bl_toks)}")
                print(f"outputs_w_bl max: {max(w_bl_toks)}")
        
        if success is False:
            print(f"Unable to get both a no_bl and w_bl output that were decodeable after {samples_taken} tries, returning empty strings.")
            example.update({"no_bl_output":""})
            example.update({"w_bl_output":""})
            if bl_processor_list:
                if bl_processor_list[0].bl_ids is not None:
                    example["bl_ids"] = []
                if bl_processor_list[0].spike_entropies is not None:
                    example["spike_entropies"] = []

    # Able to get lengths in here by checking 
    # truncated input shape versus the output shape

    example.update({
        # "baseline_num_tokens_generated" : untruncated_inputs.shape[1] - inputs.shape[1], # want this earlier now
        "no_bl_num_tokens_generated"    : outputs_no_bl.shape[1] - inputs.shape[1],
        "w_bl_num_tokens_generated"     : outputs_w_bl.shape[1] - inputs.shape[1]
    })
    example.update({
        "no_bl_sec_per_tok"             : example["no_bl_gen_time"]/example["no_bl_num_tokens_generated"],
        "no_bl_tok_per_sec"             : example["no_bl_num_tokens_generated"]/example["no_bl_gen_time"],
        "w_bl_sec_per_tok"             : example["w_bl_gen_time"]/example["w_bl_num_tokens_generated"],
        "w_bl_tok_per_sec"             : example["w_bl_num_tokens_generated"]/example["w_bl_gen_time"],
    })
    
    # now done externally because these persist outside this func
    # # remove any fields we don't need to keep
    # del example["inputs"]
    # del example["untruncated_inputs"]
    
    return example


def compute_bl_metrics(example: dict, 
                        idx: int,
                        hf_model_name: str = None,
                        tokenizer: Tokenizer=None,
                        initial_seed: int = None,
                        dynamic_seed: str = None,
                        bl_proportion: float = None,
                        use_cuda: bool = None,
                        record_hits: bool = False,
                        limit_output_tokens: int = 0):

    # if example["idx"] == 3: breakpoint()
    # okay need to catch an odd bug here and fix things
    baseline_before = example["baseline_completion"]
    example["baseline_completion"] = baseline_before.replace(example["truncated_input"][:-1],"")
    if example["baseline_completion"] != baseline_before:
        print("baseline input replacement bug occurred!")
    
    no_bl_before = example["no_bl_output"]
    example["no_bl_output"] = no_bl_before.replace(example["truncated_input"][:-1],"")
    if example["no_bl_output"] != no_bl_before:
        print("no_bl_output input replacement bug occurred!")
    
    w_bl_before = example["w_bl_output"]
    example["w_bl_output"] = w_bl_before.replace(example["truncated_input"][:-1],"")
    if example["w_bl_output"] != w_bl_before:
        print("w_bl_output input replacement bug occurred!")

    if ("w_bl_output_attacked" in example):
        w_bl_attacked_before = example["w_bl_output_attacked"]
        example["w_bl_output_attacked"] = w_bl_attacked_before.replace(example["truncated_input"][:-1],"")
        if example["w_bl_output_attacked"] != w_bl_attacked_before:
            print("w_bl_output_attacked input replacement bug occurred!")

    ##########

    # preprocess for model generation/completion
    inputs = tokenize_and_truncate({"text":example["truncated_input"]}, 
                                    completion_length=0, 
                                    hf_model_name=hf_model_name,
                                    tokenizer=tokenizer)["inputs"]

    baseline_outputs = tokenize_and_truncate({"text":example["baseline_completion"]}, 
                                        completion_length=0, 
                                        hf_model_name=hf_model_name,
                                        tokenizer=tokenizer)["inputs"][:,1:]

    no_bl_outputs = tokenize_and_truncate({"text":example["no_bl_output"]}, 
                                        completion_length=0, 
                                        hf_model_name=hf_model_name,
                                        tokenizer=tokenizer)["inputs"][:,1:]

    w_bl_outputs = tokenize_and_truncate({"text":example["w_bl_output"]}, 
                                        completion_length=0, 
                                        hf_model_name=hf_model_name,
                                        tokenizer=tokenizer)["inputs"][:,1:]
    if "w_bl_output_attacked" in example:
        w_bl_attacked_outputs = tokenize_and_truncate({"text":example["w_bl_output_attacked"]}, 
                                            completion_length=0, 
                                            hf_model_name=hf_model_name,
                                            tokenizer=tokenizer)["inputs"][:,1:]
    else:
        w_bl_attacked_outputs = None

    if limit_output_tokens > 0:
        example["orig_baseline_completion"] = example["baseline_completion"]
        example["orig_real_completion_length"] = example["real_completion_length"]
        baseline_outputs = baseline_outputs[:,:limit_output_tokens]
        example["real_completion_length"] = baseline_outputs.shape[1]
        example["baseline_completion"] = tokenizer.batch_decode(baseline_outputs, skip_special_tokens=True)[0]

        example["orig_no_bl_output"] = example["no_bl_output"]
        example["orig_no_bl_num_tokens_generated"] = example["no_bl_num_tokens_generated"]
        no_bl_outputs = no_bl_outputs[:,:limit_output_tokens]
        example["no_bl_num_tokens_generated"] = no_bl_outputs.shape[1]
        example["no_bl_output"] = tokenizer.batch_decode(no_bl_outputs, skip_special_tokens=True)[0]

        example["orig_w_bl_output"] = example["w_bl_output"]
        example["orig_w_bl_num_tokens_generated"] = example["w_bl_num_tokens_generated"]
        w_bl_outputs = w_bl_outputs[:,:limit_output_tokens]
        example["w_bl_num_tokens_generated"] = w_bl_outputs.shape[1]
        example["w_bl_output"] = tokenizer.batch_decode(w_bl_outputs, skip_special_tokens=True)[0]

        example["orig_spike_entropies"] = example["spike_entropies"]
        example["spike_entropies"] = [example["spike_entropies"][0][:limit_output_tokens]]

        if "w_bl_output_attacked" in example:
            # raise NotImplementedError("Havent thought what to do yet for this")
            example["orig_w_bl_output_attacked"] = example["w_bl_output_attacked"]
            # example["orig_w_bl_attacked_num_tokens_generated"] = example["w_bl_attacked_num_tokens_generated"]
            w_bl_attacked_outputs = w_bl_attacked_outputs[:,:limit_output_tokens]
            example["w_bl_attacked_num_tokens_generated"] = w_bl_attacked_outputs.shape[1]
            example["w_bl_output_attacked"] = tokenizer.batch_decode(w_bl_attacked_outputs, skip_special_tokens=True)[0]

    # score the 3 sequence completions/outputs wrt to bl hits
    result = score_sequence(inputs=inputs,
                            outputs=baseline_outputs, # <-- real text completions
                            initial_seed=initial_seed,
                            dynamic_seed=dynamic_seed,
                            bl_proportion=bl_proportion, 
                            tokenizer=tokenizer,
                            use_cuda=use_cuda,
                            record_hits=record_hits,
                            debug=False)
    if record_hits:
        bl_hits, num_toks_gend, hit_list = result
    else:
        bl_hits, num_toks_gend = result
    example.update({"baseline_num_toks_gend_eq_0":(num_toks_gend == 0)})
    # if num_toks_gend < 0.99*example["real_completion_length"]: breakpoint()
    # if len(hit_list) < 0.99*example["real_completion_length"]: breakpoint()

    if num_toks_gend == 0:
        # print("No tokens generated, odd, avoiding div by zero and returning -1's")
        wl_frac = -1
        bl_frac = -1
    else:
        wl_frac = (num_toks_gend-bl_hits)/num_toks_gend
        bl_frac = bl_hits/num_toks_gend
    baseline_stats = {
        "baseline_whitelist_fraction": wl_frac,
        "baseline_blacklist_fraction": bl_frac
    }
    example.update(baseline_stats)
    if record_hits: example.update({"baseline_hit_list":hit_list})
    
    result = score_sequence(inputs=inputs,
                                            outputs=no_bl_outputs, # <-- non-blacklisted version
                                            initial_seed=initial_seed,
                                            dynamic_seed=dynamic_seed,
                                            bl_proportion=bl_proportion, 
                                            tokenizer=tokenizer,
                                            record_hits=record_hits,
                                            debug=False)
    if record_hits:
        bl_hits, num_toks_gend, hit_list = result
    else:
        bl_hits, num_toks_gend = result
    example.update({"no_bl_num_toks_gend_eq_0":(num_toks_gend == 0)})
    # if num_toks_gend < 0.99*example["no_bl_num_tokens_generated"]: breakpoint()
    # if len(hit_list) < 0.99*example["no_bl_num_tokens_generated"]: breakpoint()

    if num_toks_gend == 0:
        # print("No tokens generated, odd, avoiding div by zero and returning -1's")
        wl_frac = -1
        bl_frac = -1
    else:
        wl_frac = (num_toks_gend-bl_hits)/num_toks_gend
        bl_frac = bl_hits/num_toks_gend
    no_bl_stats = {
        "no_bl_whitelist_fraction": wl_frac,
        "no_bl_blacklist_fraction": bl_frac
    }
    example.update(no_bl_stats)
    if record_hits: example.update({"no_bl_hit_list":hit_list})
    
    result = score_sequence(inputs=inputs,
                                            outputs=w_bl_outputs, # <-- blacklisted version
                                            initial_seed=initial_seed,
                                            dynamic_seed=dynamic_seed,
                                            bl_proportion=bl_proportion, 
                                            tokenizer=tokenizer,
                                            record_hits=record_hits,
                                            # breakpoint_on_hit=True, # banging head against wall
                                            debug=False)
    if record_hits:
        bl_hits, num_toks_gend, hit_list = result
    else:
        bl_hits, num_toks_gend = result
    example.update({"w_bl_num_toks_gend_eq_0":(num_toks_gend == 0)})
    # if num_toks_gend < 0.99*example["w_bl_num_tokens_generated"]: breakpoint()
    # if len(hit_list) < 0.99*example["w_bl_num_tokens_generated"]: breakpoint()

    if num_toks_gend == 0:
        # print("No tokens generated, odd, avoiding div by zero and returning -1's")
        wl_frac = -1
        bl_frac = -1
    else:
        wl_frac = (num_toks_gend-bl_hits)/num_toks_gend
        bl_frac = bl_hits/num_toks_gend
    w_bl_stats = {
        "w_bl_whitelist_fraction": wl_frac,
        "w_bl_blacklist_fraction": bl_frac
    }
    example.update(w_bl_stats)
    if record_hits: example.update({"w_bl_hit_list":hit_list})

    if w_bl_attacked_outputs is not None:
        result = score_sequence(inputs=inputs,
                                outputs=w_bl_attacked_outputs, # <-- blacklisted but attacked version
                                initial_seed=initial_seed,
                                dynamic_seed=dynamic_seed,
                                bl_proportion=bl_proportion, 
                                tokenizer=tokenizer,
                                record_hits=record_hits,
                                # breakpoint_on_hit=True, # banging head against wall
                                debug=False)
        if record_hits:
            bl_hits, num_toks_gend, hit_list = result
        else:
            bl_hits, num_toks_gend = result
        example.update({"w_bl_attacked_num_toks_gend_eq_0":(num_toks_gend == 0)})
        # if (num_toks_gend-bl_hits)/(num_toks_gend) < 1.0: breakpoint()

        if num_toks_gend == 0:
            # print("No tokens generated, odd, avoiding div by zero and returning -1's")
            wl_frac = -1
            bl_frac = -1
        else:
            wl_frac = (num_toks_gend-bl_hits)/num_toks_gend
            bl_frac = bl_hits/num_toks_gend
        w_bl_attacked_stats = {
            "w_bl_attacked_num_tokens_generated": num_toks_gend,
            "w_bl_attacked_whitelist_fraction": wl_frac,
            "w_bl_attacked_blacklist_fraction": bl_frac
        }
        example.update(w_bl_attacked_stats)
        if record_hits: example.update({"w_bl_attacked_hit_list":hit_list})
    
    return example



def aggregate_bl_stats(example: dict, idx: int, stat_table: dict):
    
    stat_table["baseline_stats"]["whitelist_fraction"] += example["baseline_stats"]["whitelist_fraction"]
    stat_table["baseline_stats"]["blacklist_fraction"] += example["baseline_stats"]["blacklist_fraction"]
    
    stat_table["w_bl_stats"]["whitelist_fraction"] += example["w_bl_stats"]["whitelist_fraction"]
    stat_table["w_bl_stats"]["blacklist_fraction"] += example["w_bl_stats"]["blacklist_fraction"]

    stat_table["no_bl_stats"]["whitelist_fraction"] += example["no_bl_stats"]["whitelist_fraction"]
    stat_table["no_bl_stats"]["blacklist_fraction"] += example["no_bl_stats"]["blacklist_fraction"]

    stat_table["num_examples"] += 1

    return example


def compute_ppl_single(prefix_and_output_text = None,
                        output_text = None,
                        oracle_model_name = None,
                        oracle_model = None,
                        oracle_tokenizer = None):

    with torch.no_grad():
        tokd_prefix = tokenize_and_truncate({"text":prefix_and_output_text}, completion_length=0, hf_model_name=oracle_model_name, tokenizer=oracle_tokenizer, model_max_seq_len=oracle_model.config.max_position_embeddings)["inputs"]
        tokd_inputs = tokd_prefix
        # if only want to score the "generation" part we need the suffix tokenization length
        tokd_suffix = tokenize_and_truncate({"text":output_text}, completion_length=0, hf_model_name=oracle_model_name, tokenizer=oracle_tokenizer, model_max_seq_len=oracle_model.config.max_position_embeddings)["inputs"]

        tokd_inputs = tokd_inputs.to(oracle_model.device)
        # make labels, mark if not including all positions
        tokd_labels = tokd_inputs.clone().detach()
        tokd_labels[:,:tokd_labels.shape[1]-tokd_suffix.shape[1]+1] = -100

        outputs = oracle_model(input_ids=tokd_inputs, labels=tokd_labels)
        loss = outputs.loss # avg CE loss all positions (except -100, TODO plz check that this is working correctly)
        ppl = torch.tensor(math.exp(loss))
    
    return loss.item(), ppl.item()


def evaluate_generation_fluency(example: dict, 
                                idx: int,
                                oracle_model_name = None,
                                oracle_model = None,
                                oracle_tokenizer = None):

    # pull out the required fields from the pipeline results
    inputs_plus_baseline_output = f"{example['truncated_input']}{example['baseline_completion']}"
    baseline_output = f"{example['baseline_completion']}"

    inputs_plus_no_bl_output = f"{example['truncated_input']}{example['no_bl_output']}"
    no_bl_output = f"{example['no_bl_output']}"

    inputs_plus_w_bl_output = f"{example['truncated_input']}{example['w_bl_output']}"
    w_bl_output = f"{example['w_bl_output']}"

    # add metrics
    loss, ppl = compute_ppl_single(inputs_plus_baseline_output, baseline_output, oracle_model_name, oracle_model, oracle_tokenizer)
    example["baseline_loss"] = loss
    example["baseline_ppl"] = ppl
    loss, ppl = compute_ppl_single(inputs_plus_no_bl_output, no_bl_output, oracle_model_name, oracle_model, oracle_tokenizer)
    example["no_bl_loss"] = loss
    example["no_bl_ppl"] = ppl
    loss, ppl = compute_ppl_single(inputs_plus_w_bl_output, w_bl_output, oracle_model_name, oracle_model, oracle_tokenizer)
    example["w_bl_loss"] = loss
    example["w_bl_ppl"] = ppl

    # del any temp values
    return example


def add_idx(example,idx):
    example.update({"idx":idx})
    return example


def check_input_lengths(example,idx, min_sample_len=0, min_prompt_len=0, min_completion_len=0):
    orig_sample_length = example["orig_sample_length"]
    prompt_length = example["prompt_length"]
    real_completion_length = example["real_completion_length"]

    # breakpoint()

    conds = all([
        orig_sample_length >= min_sample_len,
        prompt_length >= min_prompt_len,
        real_completion_length >= min_completion_len,
    ])
    return conds


def check_output_lengths(example,min_output_len=0):
    no_bl_output_len = example["no_bl_num_tokens_generated"]
    w_bl_output_len = example["w_bl_num_tokens_generated"]
    conds = all([
        no_bl_output_len >= min_output_len,
        w_bl_output_len >= min_output_len,
    ])
    return conds


