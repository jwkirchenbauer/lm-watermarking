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

import os
import argparse
from argparse import Namespace
from pprint import pprint
from functools import partial

import numpy # for gradio hot reload
import gradio as gr

import torch

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)

from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")

    parser.add_argument(
        "--run_gradio",
        type=str2bool,
        default=True,
        help="Whether to launch as a gradio demo. Set to False if not installed and want to just run the stdout version.",
    )
    parser.add_argument(
        "--demo_public",
        type=str2bool,
        default=False,
        help="Whether to expose the gradio demo to the internet.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-6.7b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=123,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--skip_model_load",
        type=str2bool,
        default=False,
        help="Skip the model loading to debug the interface.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
        help="Whether to run model in float16 precsion.",
    )
    args = parser.parse_args()
    return args

def load_model(args):
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5","T0"]])
    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt","opt","bloom"]])
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.float16, device_map='auto')
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    return model, tokenizer, device

def generate(prompt, args, model=None, device=None, tokenizer=None):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
       and generate watermarked text by passing it to the generate method of the model
       as a logits processor. """
    
    print(f"Generating with {args}")

    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=args.gamma,
                                                    delta=args.delta,
                                                    seeding_scheme=args.seeding_scheme,
                                                    select_green_tokens=args.select_green_tokens)

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))

    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )
    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]), 
        **gen_kwargs
    )
    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    else:
        args.prompt_max_length = 2048-args.max_new_tokens

    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(**tokd_input)

    # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
    if args.seed_separately: 
        torch.manual_seed(args.generation_seed)
    output_with_watermark = generate_with_watermark(**tokd_input)

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:,tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark, 
            decoded_output_with_watermark,
            args) 
            # decoded_output_with_watermark)

def format_names(s):
    """Format names for the gradio demo interface"""
    s=s.replace("num_tokens_scored","Tokens Counted (T)")
    s=s.replace("num_green_tokens","# Tokens in Greenlist")
    s=s.replace("green_fraction","Fraction of T in Greenlist")
    s=s.replace("z_score","z-score")
    s=s.replace("p_value","p value")
    s=s.replace("prediction","Prediction")
    s=s.replace("confidence","Confidence")
    return s

def list_format_scores(score_dict, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    # lst_2d.append(["z-score threshold", f"{detection_threshold}"])
    for k,v in score_dict.items():
        if k=='green_fraction': 
            lst_2d.append([format_names(k), f"{v:.1%}"])
        elif k=='confidence': 
            lst_2d.append([format_names(k), f"{v:.3%}"])
        elif isinstance(v, float): 
            lst_2d.append([format_names(k), f"{v:.3g}"])
        elif isinstance(v, bool):
            lst_2d.append([format_names(k), ("Watermarked" if v else "Human/Unwatermarked")])
        else: 
            lst_2d.append([format_names(k), f"{v}"])
    if "confidence" in score_dict:
        lst_2d.insert(-2,["z-score Threshold", f"{detection_threshold}"])
    else:
        lst_2d.insert(-1,["z-score Threshold", f"{detection_threshold}"])
    return lst_2d

def detect(input_text, args, device=None, tokenizer=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=args.gamma,
                                        seeding_scheme=args.seeding_scheme,
                                        device=device,
                                        tokenizer=tokenizer,
                                        z_threshold=args.detection_z_threshold,
                                        normalizers=args.normalizers,
                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                        select_green_tokens=args.select_green_tokens)
    if len(input_text)-1 > watermark_detector.min_prefix_len:
        score_dict = watermark_detector.detect(input_text)
        # output = str_format_scores(score_dict, watermark_detector.z_threshold)
        output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        # output = (f"Error: string not long enough to compute watermark presence.")
        output = [["Error","string too short to compute metrics"]]
        output += [["",""] for _ in range(6)]
    return output, args

def run_gradio(args, model=None, device=None, tokenizer=None):
    """Define and launch the gradio demo interface"""
    generate_partial = partial(generate, model=model, device=device, tokenizer=tokenizer)
    detect_partial = partial(detect, device=device, tokenizer=tokenizer)

    with gr.Blocks() as demo:
        # Top section, greeting and instructions
        with gr.Row():
            with gr.Column(scale=9):
                gr.Markdown(
                """
                ## 💧 [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226) 🔍
                """
                )
            with gr.Column(scale=1):
                gr.Markdown(
                """
                [![](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/jwkirchenbauer/lm-watermarking)
                """
                )
            # with gr.Column(scale=2):
            #     pass
            # ![visitor badge](https://visitor-badge.glitch.me/badge?page_id=tomg-group-umd_lm-watermarking) # buggy

        with gr.Accordion("Understanding the output metrics",open=False):
            gr.Markdown(
            """
            - `z-score threshold` : The cuttoff for the hypothesis test
            - `Tokens Counted (T)` : The number of tokens in the output that were counted by the detection algorithm. 
                The first token is ommitted in the simple, single token seeding scheme since there is no way to generate
                a greenlist for it as it has no prefix token(s). Under the "Ignore Bigram Repeats" detection algorithm, 
                described in the bottom panel, this can be much less than the total number of tokens generated if there is a lot of repetition.
            - `# Tokens in Greenlist` : The number of tokens that were observed to fall in their respective greenlist
            - `Fraction of T in Greenlist` : The `# Tokens in Greenlist` / `T`. This is expected to be approximately `gamma` for human/unwatermarked text.
            - `z-score` : The test statistic for the detection hypothesis test. If larger than the `z-score threshold` 
                we "reject the null hypothesis" that the text is human/unwatermarked, and conclude it is watermarked
            - `p value` : The likelihood of observing the computed `z-score` under the null hypothesis. This is the likelihood of 
                observing the `Fraction of T in Greenlist` given that the text was generated without knowledge of the watermark procedure/greenlists.
                If this is extremely _small_ we are confident that this many green tokens was not chosen by random chance.
            -  `prediction` : The outcome of the hypothesis test - whether the observed `z-score` was higher than the `z-score threshold`
            - `confidence` : If we reject the null hypothesis, and the `prediction` is "Watermarked", then we report 1-`p value` to represent 
                the confidence of the detection based on the unlikeliness of this `z-score` observation.
            """
            )

        with gr.Accordion("A note on model capability",open=True):
            gr.Markdown(
                """
                This demo uses open-source language models that fit on a single GPU. These models are less powerful than proprietary commercial tools like ChatGPT, Claude, or Bard. 

                Importantly, we use a language model that is designed to "complete" your prompt, and not a model this is fine-tuned to follow instructions. 
                For best results, prompt the model with a few sentences that form the beginning of a paragraph, and then allow it to "continue" your paragraph. 
                Some examples include the opening paragraph of a wikipedia article, or the first few sentences of a story. 
                Longer prompts that end mid-sentence will result in more fluent generations.
                """
                )
        gr.Markdown(f"Language model: {args.model_name_or_path} {'(float16 mode)' if args.load_fp16 else ''}")

        # Construct state for parameters, define updates and toggles
        default_prompt = args.__dict__.pop("default_prompt")
        session_args = gr.State(value=args)

        with gr.Tab("Generate and Detect"):
            
            with gr.Row():
                prompt = gr.Textbox(label=f"Prompt", interactive=True,lines=10,max_lines=10, value=default_prompt)
            with gr.Row():
                generate_btn = gr.Button("Generate")
            with gr.Row():
                with gr.Column(scale=2):
                    output_without_watermark = gr.Textbox(label="Output Without Watermark", interactive=False,lines=14,max_lines=14)
                with gr.Column(scale=1):
                    # without_watermark_detection_result = gr.Textbox(label="Detection Result", interactive=False,lines=14,max_lines=14)
                    without_watermark_detection_result = gr.Dataframe(headers=["Metric", "Value"], interactive=False,row_count=7,col_count=2)
            with gr.Row():
                with gr.Column(scale=2):
                    output_with_watermark = gr.Textbox(label="Output With Watermark", interactive=False,lines=14,max_lines=14)
                with gr.Column(scale=1):
                    # with_watermark_detection_result = gr.Textbox(label="Detection Result", interactive=False,lines=14,max_lines=14)
                    with_watermark_detection_result = gr.Dataframe(headers=["Metric", "Value"],interactive=False,row_count=7,col_count=2)

            redecoded_input = gr.Textbox(visible=False)
            truncation_warning = gr.Number(visible=False)
            def truncate_prompt(redecoded_input, truncation_warning, orig_prompt, args):
                if truncation_warning:
                    return redecoded_input + f"\n\n[Prompt was truncated before generation due to length...]", args
                else: 
                    return orig_prompt, args
        
        with gr.Tab("Detector Only"):
            with gr.Row():
                with gr.Column(scale=2):
                    detection_input = gr.Textbox(label="Text to Analyze", interactive=True,lines=14,max_lines=14)
                with gr.Column(scale=1):
                    # detection_result = gr.Textbox(label="Detection Result", interactive=False,lines=14,max_lines=14)
                    detection_result = gr.Dataframe(headers=["Metric", "Value"], interactive=False,row_count=7,col_count=2)
            with gr.Row():
                    detect_btn = gr.Button("Detect")

        # Parameter selection group
        with gr.Accordion("Advanced Settings",open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"#### Generation Parameters")
                    with gr.Row():
                        decoding = gr.Radio(label="Decoding Method",choices=["multinomial", "greedy"], value=("multinomial" if args.use_sampling else "greedy"))
                    with gr.Row():
                        sampling_temp = gr.Slider(label="Sampling Temperature", minimum=0.1, maximum=1.0, step=0.1, value=args.sampling_temp, visible=True)
                    with gr.Row():
                        generation_seed = gr.Number(label="Generation Seed",value=args.generation_seed, interactive=True)
                    with gr.Row():
                        n_beams = gr.Dropdown(label="Number of Beams",choices=list(range(1,11,1)), value=args.n_beams, visible=(not args.use_sampling))
                    with gr.Row():
                        max_new_tokens = gr.Slider(label="Max Generated Tokens", minimum=10, maximum=1000, step=10, value=args.max_new_tokens)

                with gr.Column(scale=1):
                    gr.Markdown(f"#### Watermark Parameters")
                    with gr.Row():
                        gamma = gr.Slider(label="gamma",minimum=0.1, maximum=0.9, step=0.05, value=args.gamma)
                    with gr.Row():
                        delta = gr.Slider(label="delta",minimum=0.0, maximum=10.0, step=0.1, value=args.delta)
                    gr.Markdown(f"#### Detector Parameters")
                    with gr.Row():
                        detection_z_threshold = gr.Slider(label="z-score threshold",minimum=0.0, maximum=10.0, step=0.1, value=args.detection_z_threshold)
                    with gr.Row():
                        ignore_repeated_bigrams = gr.Checkbox(label="Ignore Bigram Repeats")
                    with gr.Row():
                        normalizers = gr.CheckboxGroup(label="Normalizations", choices=["unicode", "homoglyphs", "truecase"], value=args.normalizers)
            # with gr.Accordion("Actual submitted parameters:",open=False):
            with gr.Row():
                gr.Markdown(f"_Note: sliders don't always update perfectly. Clicking on the bar or using the number window to the right can help. Window below shows the current settings._")
            with gr.Row():
                current_parameters = gr.Textbox(label="Current Parameters", value=args)
            with gr.Accordion("Legacy Settings",open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        seed_separately = gr.Checkbox(label="Seed both generations separately", value=args.seed_separately)
                    with gr.Column(scale=1):
                        select_green_tokens = gr.Checkbox(label="Select 'greenlist' from partition", value=args.select_green_tokens)
        
        with gr.Accordion("Understanding the settings",open=False):
            gr.Markdown(
            """
            #### Generation Parameters:

            - Decoding Method : We can generate tokens from the model using either multinomial sampling or we can generate using greedy decoding.
            - Sampling Temperature : If using multinomial sampling we can set the temperature of the sampling distribution. 
                                0.0 is equivalent to greedy decoding, and 1.0 is the maximum amount of variability/entropy in the next token distribution.
                                0.7 strikes a nice balance between faithfulness to the model's estimate of top candidates while adding variety. Does not apply for greedy decoding.
            - Generation Seed : The integer to pass to the torch random number generator before running generation. Makes the multinomial sampling strategy
                                outputs reproducible. Does not apply for greedy decoding.
            - Number of Beams : When using greedy decoding, we can also set the number of beams to > 1 to enable beam search. 
                                This is not implemented/excluded from paper for multinomial sampling but may be added in future.
            - Max Generated Tokens : The `max_new_tokens` parameter passed to the generation method to stop the output at a certain number of new tokens. 
                                    Note that the model is free to generate fewer tokens depending on the prompt. 
                                    Implicitly this sets the maximum number of prompt tokens possible as the model's maximum input length minus `max_new_tokens`,
                                    and inputs will be truncated accordingly.
            
            #### Watermark Parameters:

            - gamma : The fraction of the vocabulary to be partitioned into the greenlist at each generation step. 
                     Smaller gamma values create a stronger watermark by enabling the watermarked model to achieve 
                     a greater differentiation from human/unwatermarked text because it is preferentially sampling 
                     from a smaller green set making those tokens less likely to occur by chance.
            - delta : The amount of positive bias to add to the logits of every token in the greenlist 
                        at each generation step before sampling/choosing the next token. Higher delta values 
                        mean that the greenlist tokens are more heavily preferred by the watermarked model
                        and as the bias becomes very large the watermark transitions from "soft" to "hard". 
                        For a hard watermark, nearly all tokens are green, but this can have a detrimental effect on
                        generation quality, especially when there is not a lot of flexibility in the distribution.

            #### Detector Parameters:
            
            - z-score threshold : the z-score cuttoff for the hypothesis test. Higher thresholds (such as 4.0) make
                                _false positives_ (predicting that human/unwatermarked text is watermarked) very unlikely
                                as a genuine human text with a significant number of tokens will almost never achieve 
                                that high of a z-score. Lower thresholds will capture more _true positives_ as some watermarked
                                texts will contain less green tokens and achive a lower z-score, but still pass the lower bar and 
                                be flagged as "watermarked". However, a lowere threshold will increase the chance that human text 
                                that contains a slightly higher than average number of green tokens is erroneously flagged. 
                                4.0-5.0 offers extremely low false positive rates while still accurately catching most watermarked text.
            - Ignore Bigram Repeats : This alternate detection algorithm only considers the unique bigrams in the text during detection, 
                                    computing the greenlists based on the first in each pair and checking whether the second falls within the list.
                                    This means that `T` is now the unique number of bigrams in the text, which becomes less than the total
                                    number of tokens generated if the text contains a lot of repetition. See the paper for a more detailed discussion.
            - Normalizations : we implement a few basic normaliations to defend against various adversarial perturbations of the
                                text analyzed during detection. Currently we support converting all chracters to unicode, 
                                replacing homoglyphs with a canonical form, and standardizing the capitalization. 
                                See the paper for a detailed discussion of input normalization. 
            """
            )
        
        gr.HTML("""
                <p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. 
                    Follow the github link at the top and host the demo on your own GPU hardware to test out larger models.
                <br/>
                <a href="https://huggingface.co/spaces/tomg-group-umd/lm-watermarking?duplicate=true">
                <img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
                <p/>
                """)
        
        # Register main generation tab click, outputing generations as well as a the encoded+redecoded+potentially truncated prompt and flag
        generate_btn.click(fn=generate_partial, inputs=[prompt,session_args], outputs=[redecoded_input, truncation_warning, output_without_watermark, output_with_watermark,session_args])
        # Show truncated version of prompt if truncation occurred
        redecoded_input.change(fn=truncate_prompt, inputs=[redecoded_input,truncation_warning,prompt,session_args], outputs=[prompt,session_args])
        # Call detection when the outputs (of the generate function) are updated
        output_without_watermark.change(fn=detect_partial, inputs=[output_without_watermark,session_args], outputs=[without_watermark_detection_result,session_args])
        output_with_watermark.change(fn=detect_partial, inputs=[output_with_watermark,session_args], outputs=[with_watermark_detection_result,session_args])
        # Register main detection tab click
        detect_btn.click(fn=detect_partial, inputs=[detection_input,session_args], outputs=[detection_result, session_args])

        # State management logic
        # update callbacks that change the state dict
        def update_sampling_temp(session_state, value): session_state.sampling_temp = float(value); return session_state
        def update_generation_seed(session_state, value): session_state.generation_seed = int(value); return session_state
        def update_gamma(session_state, value): session_state.gamma = float(value); return session_state
        def update_delta(session_state, value): session_state.delta = float(value); return session_state
        def update_detection_z_threshold(session_state, value): session_state.detection_z_threshold = float(value); return session_state
        def update_decoding(session_state, value):
            if value == "multinomial":
                session_state.use_sampling = True
            elif value == "greedy":
                session_state.use_sampling = False
            return session_state
        def toggle_sampling_vis(value):
            if value == "multinomial":
                return gr.update(visible=True)
            elif value == "greedy":
                return gr.update(visible=False)
        def toggle_sampling_vis_inv(value):
            if value == "multinomial":
                return gr.update(visible=False)
            elif value == "greedy":
                return gr.update(visible=True)
        def update_n_beams(session_state, value): session_state.n_beams = value; return session_state
        def update_max_new_tokens(session_state, value): session_state.max_new_tokens = int(value); return session_state
        def update_ignore_repeated_bigrams(session_state, value): session_state.ignore_repeated_bigrams = value; return session_state
        def update_normalizers(session_state, value): session_state.normalizers = value; return session_state
        def update_seed_separately(session_state, value): session_state.seed_separately = value; return session_state
        def update_select_green_tokens(session_state, value): session_state.select_green_tokens = value; return session_state
        # registering callbacks for toggling the visibilty of certain parameters
        decoding.change(toggle_sampling_vis,inputs=[decoding], outputs=[sampling_temp])
        decoding.change(toggle_sampling_vis,inputs=[decoding], outputs=[generation_seed])
        decoding.change(toggle_sampling_vis_inv,inputs=[decoding], outputs=[n_beams])
        # registering all state update callbacks
        decoding.change(update_decoding,inputs=[session_args, decoding], outputs=[session_args])
        sampling_temp.change(update_sampling_temp,inputs=[session_args, sampling_temp], outputs=[session_args])
        generation_seed.change(update_generation_seed,inputs=[session_args, generation_seed], outputs=[session_args])
        n_beams.change(update_n_beams,inputs=[session_args, n_beams], outputs=[session_args])
        max_new_tokens.change(update_max_new_tokens,inputs=[session_args, max_new_tokens], outputs=[session_args])
        gamma.change(update_gamma,inputs=[session_args, gamma], outputs=[session_args])
        delta.change(update_delta,inputs=[session_args, delta], outputs=[session_args])
        detection_z_threshold.change(update_detection_z_threshold,inputs=[session_args, detection_z_threshold], outputs=[session_args])
        ignore_repeated_bigrams.change(update_ignore_repeated_bigrams,inputs=[session_args, ignore_repeated_bigrams], outputs=[session_args])
        normalizers.change(update_normalizers,inputs=[session_args, normalizers], outputs=[session_args])
        seed_separately.change(update_seed_separately,inputs=[session_args, seed_separately], outputs=[session_args])
        select_green_tokens.change(update_select_green_tokens,inputs=[session_args, select_green_tokens], outputs=[session_args])
        # register additional callback on button clicks that updates the shown parameters window
        generate_btn.click(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        detect_btn.click(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        # When the parameters change, display the update and fire detection, since some detection params dont change the model output.
        gamma.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        gamma.change(fn=detect_partial, inputs=[output_without_watermark,session_args], outputs=[without_watermark_detection_result,session_args])
        gamma.change(fn=detect_partial, inputs=[output_with_watermark,session_args], outputs=[with_watermark_detection_result,session_args])
        gamma.change(fn=detect_partial, inputs=[detection_input,session_args], outputs=[detection_result,session_args])
        detection_z_threshold.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        detection_z_threshold.change(fn=detect_partial, inputs=[output_without_watermark,session_args], outputs=[without_watermark_detection_result,session_args])
        detection_z_threshold.change(fn=detect_partial, inputs=[output_with_watermark,session_args], outputs=[with_watermark_detection_result,session_args])
        detection_z_threshold.change(fn=detect_partial, inputs=[detection_input,session_args], outputs=[detection_result,session_args])
        ignore_repeated_bigrams.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        ignore_repeated_bigrams.change(fn=detect_partial, inputs=[output_without_watermark,session_args], outputs=[without_watermark_detection_result,session_args])
        ignore_repeated_bigrams.change(fn=detect_partial, inputs=[output_with_watermark,session_args], outputs=[with_watermark_detection_result,session_args])
        ignore_repeated_bigrams.change(fn=detect_partial, inputs=[detection_input,session_args], outputs=[detection_result,session_args])
        normalizers.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        normalizers.change(fn=detect_partial, inputs=[output_without_watermark,session_args], outputs=[without_watermark_detection_result,session_args])
        normalizers.change(fn=detect_partial, inputs=[output_with_watermark,session_args], outputs=[with_watermark_detection_result,session_args])
        normalizers.change(fn=detect_partial, inputs=[detection_input,session_args], outputs=[detection_result,session_args])
        select_green_tokens.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        select_green_tokens.change(fn=detect_partial, inputs=[output_without_watermark,session_args], outputs=[without_watermark_detection_result,session_args])
        select_green_tokens.change(fn=detect_partial, inputs=[output_with_watermark,session_args], outputs=[with_watermark_detection_result,session_args])
        select_green_tokens.change(fn=detect_partial, inputs=[detection_input,session_args], outputs=[detection_result,session_args])


    demo.queue(concurrency_count=3)

    if args.demo_public:
        demo.launch(share=True) # exposes app to the internet via randomly generated link
    else:
        demo.launch()

def main(args): 
    """Run a command line version of the generation and detection operations
        and optionally launch and serve the gradio demo"""
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    print(args)

    if not args.skip_model_load:
        model, tokenizer, device = load_model(args)
    else:
        model, tokenizer, device = None, None, None

    # Generate and detect, report to stdout
    if not args.skip_model_load:
        input_text = (
        "The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a "
        "species of turtle native to the brackish coastal tidal marshes of the "
        "Northeastern and southern United States, and in Bermuda.[6] It belongs "
        "to the monotypic genus Malaclemys. It has one of the largest ranges of "
        "all turtles in North America, stretching as far south as the Florida Keys "
        "and as far north as Cape Cod.[7] The name 'terrapin' is derived from the "
        "Algonquian word torope.[8] It applies to Malaclemys terrapin in both "
        "British English and American English. The name originally was used by "
        "early European settlers in North America to describe these brackish-water "
        "turtles that inhabited neither freshwater habitats nor the sea. It retains "
        "this primary meaning in American English.[8] In British English, however, "
        "other semi-aquatic turtle species, such as the red-eared slider, might "
        "also be called terrapins. The common name refers to the diamond pattern "
        "on top of its shell (carapace), but the overall pattern and coloration "
        "vary greatly. The shell is usually wider at the back than in the front, "
        "and from above it appears wedge-shaped. The shell coloring can vary "
        "from brown to grey, and its body color can be grey, brown, yellow, "
        "or white. All have a unique pattern of wiggly, black markings or spots "
        "on their body and head. The diamondback terrapin has large webbed "
        "feet.[9] The species is"
        )

        args.default_prompt = input_text

        term_width = 80
        print("#"*term_width)
        print("Prompt:")
        print(input_text)

        _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ = generate(input_text, 
                                                                                            args, 
                                                                                            model=model, 
                                                                                            device=device, 
                                                                                            tokenizer=tokenizer)
        without_watermark_detection_result = detect(decoded_output_without_watermark, 
                                                    args, 
                                                    device=device, 
                                                    tokenizer=tokenizer)
        with_watermark_detection_result = detect(decoded_output_with_watermark, 
                                                 args, 
                                                 device=device, 
                                                 tokenizer=tokenizer)

        print("#"*term_width)
        print("Output without watermark:")
        print(decoded_output_without_watermark)
        print("-"*term_width)
        print(f"Detection result @ {args.detection_z_threshold}:")
        pprint(without_watermark_detection_result)
        print("-"*term_width)

        print("#"*term_width)
        print("Output with watermark:")
        print(decoded_output_with_watermark)
        print("-"*term_width)
        print(f"Detection result @ {args.detection_z_threshold}:")
        pprint(with_watermark_detection_result)
        print("-"*term_width)


    # Launch the app to generate and detect interactively (implements the hf space demo)
    if args.run_gradio:
        run_gradio(args, model=model, tokenizer=tokenizer, device=device)

    return

if __name__ == "__main__":

    args = parse_args()
    print(args)

    main(args)