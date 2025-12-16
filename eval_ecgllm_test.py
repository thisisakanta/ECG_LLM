#!/usr/bin/env python
# coding=utf-8
from typing import Dict, List, Optional
import argparse
import logging
import math
import os
import random
import datasets
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
# from modeling_ecg_llama import LlamaForCausalLM, LlamaModel
import transformers
from torchvision.transforms import transforms
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    # DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
    LlamaConfig
)
from ecg_data_collator import DataCollatorForSeq2Seq
# from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

import wfdb
import numpy as np
from pathlib import Path
from ecg_llm import ECG_LLM_ForCausalLM
from finetune_ecgllm_with_lora_ptbxl import encode_with_prompt_completion_format, encode_with_messages_format
from utils_ecg import get_rouge_n_gram
from eval_metrics import compute_metric_scores
from accelerate import PartialState
import logging

state=PartialState()
logger = get_logger(__name__)



def main(args):

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs], **accelerator_log_kwargs)
    accelerator = Accelerator(gradient_accumulation_steps=8, **accelerator_log_kwargs)

    # accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()

    # if args.seed is not None:
    #     set_seed(args.seed)

    logger.info("Loading the ECG_LLM model...")

    if args.model_name_or_path:
        #config = AutoConfig.from_pretrained(args.model_name_or_path)
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    
    if args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer,padding_side='left')
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.ecg_layer_idxs == 'all':
            ecg_layer_idxs = [idx for idx in range(config.num_hidden_layers)]
    else:
        ecg_layer_idxs = [str(idx) for idx in args.ecg_layer_idxs.split(",")]

    print('############### the ecg layers include ################: ', ecg_layer_idxs)

    model = ECG_LLM_ForCausalLM(args=args,
                                      model_name_or_path=args.model_name_or_path, 
                                      config=config, 
                                      tokenizer=tokenizer,
                                      ecg_layer_idxs=ecg_layer_idxs,
                                      accelerator=accelerator
                                      )

                                    
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    logger.info("Loading the ECG_LLM model success ######################")

    ############################## start to preprocess test data ###############

    data_files = {}
    dataset_args = {}
    if args.dataset_name is not None: # split the train and valid ECG data here 
        data_files["train"] = args.dataset_name
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        **dataset_args,
    )

    ######################### split the train and test dataset here ##################################

    if args.dev_ratio > 1e-6:
        train_test_split_dataset = raw_datasets['train'].train_test_split(args.dev_ratio)

    
    
    # train_dataset = train_test_split_dataset['train']
    # test_dataset = train_test_split_dataset['test'] # 

    # Preprocessing the datasets.
    if "prompt" in train_test_split_dataset['test'].column_names and "completion" in train_test_split_dataset['test'].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    elif "messages" in train_test_split_dataset['test'].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    else:
        raise ValueError("You need to have either 'prompt'&'completion' or 'messages' in your column names.")
    
    # with accelerator.main_process_first():  ######### deal with the labels and ecg data #############
    lm_datasets = train_test_split_dataset.map(
        encode_function,
        batched=False,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        remove_columns=[name for name in train_test_split_dataset["test"].column_names if name not in ["input_ids", "labels", "attention_mask", "ecg"]],
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

    # if args.dev_ratio > 1e-6:
    #     split_datasets = lm_datasets["train"].train_test_split(test_size=args.dev_ratio)
    
    # test_dataset = split_datasets["test"]

    # # Log a few random samples from the training set:
    # for index in random.sample(range(len(lm_datasets)), 3):
    #     logger.info(f"Sample {index} of the training set: {lm_datasets[index]}.")

    test_dataset = lm_datasets['test']
    test_dataloader = DataLoader(
        test_dataset, 
        shuffle=False, 
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model.llm_model, padding="longest"), ########## modified here ###############
        batch_size=args.per_device_train_batch_size
    )

    ###########################################################################################

    logger.info("Start to eval ECG_LLM model test data ######################")

    iter_batch = 0
    eval_bleu_1 = 0
    eval_bleu_2 = 0
    eval_bleu_3 = 0
    eval_bleu_4 = 0

    eval_rouge_1 = 0
    eval_rouge_2 = 0
    eval_rouge_l = 0

    eval_meteor = 0
    eval_cider = 0

    with torch.no_grad():

        for num_idx, batch in tqdm(enumerate(test_dataloader)):

            iter_batch += 1

            # if model.device.type == "cuda":

            input_ids = batch['input_ids'].cuda()
            reference_ids = input_ids.clone()
            labels = batch['labels'].cuda()
            ecg = batch['ecg'].cuda()

            mask = labels.eq(-100)
            masked = []
            for i in range(input_ids.size(0)):
                sel = input_ids[i][mask[i]]
                masked.append(sel)

            # Pad each sequence to the same length
            from torch.nn.utils.rnn import pad_sequence
            input_ids_for_inference = pad_sequence(masked, batch_first=True, padding_value=0).cuda()

            # ###  #foir debug purpose only we have used it here
            # # `input_ids_for_inference` is what you pass to generate
            # print("device:", input_ids_for_inference.device)
            # print("dtype:", input_ids_for_inference.dtype)
            # print("shape:", input_ids_for_inference.shape)

            # # max token id in batch
            # max_id = int(input_ids_for_inference.max().cpu().item())
            # print("max token id:", max_id)

            # # tokenizer / vocab info
            # print("tokenizer vocab size (len):", len(tokenizer))
            # print("tokenizer.pad_token_id:", tokenizer.pad_token_id)
            # print("tokenizer.eos_token_id:", tokenizer.eos_token_id)
            # print("tokenizer.special_tokens_map:", tokenizer.special_tokens_map)

            # # model embeddings size
            # try:
            #     emb_size = model.llm_model.get_input_embeddings().weight.shape[0]
            # except Exception:
            #     emb_size = model.get_input_embeddings().weight.shape[0]
            # print("model embedding vocab size:", emb_size)

            # # sanity: any id >= embedding size?
            # if max_id >= emb_size:
            #     print(">> ERROR: found token id >= embedding size (out of range).")
            # else:
            #     print("token ids within range.")



            ###

            output_ids = model.generate(
                input_ids=input_ids_for_inference,
                ecg = ecg,
            )
            # check here:
            generate_reports = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            reference_reports = tokenizer.batch_decode(reference_ids, skip_special_tokens=True)


            # bleu_1, bleu_2, bleu_3, bleu_4, rouge_1, rouge_2, rouge_l, meteor, cider
            nlg_metrics = ["bleu", "meteor", "rouge", "cider"]
            language_model_scores = compute_metric_scores(nlg_metrics, generate_reports, reference_reports, train_test_split_dataset['test'])
            eval_bleu_1 += language_model_scores['bleu_1']
            eval_bleu_2 += language_model_scores['bleu_2']
            eval_bleu_3 += language_model_scores['bleu_3']
            eval_bleu_4 += language_model_scores['bleu_4']

            eval_rouge_1 += language_model_scores['rouge_1']
            eval_rouge_2 += language_model_scores['rouge_2']
            #eval_rouge_l += language_model_scores['rouge_l']

            eval_meteor += language_model_scores['meteor']
            eval_cider += language_model_scores['cider']
    
    avg_bleu_1 = eval_bleu_1 / iter_batch
    avg_bleu_2 = eval_bleu_2 / iter_batch
    avg_bleu_3 = eval_bleu_3 / iter_batch
    avg_bleu_4 = eval_bleu_4 / iter_batch

    avg_rouge_1 = eval_rouge_1 / iter_batch
    avg_rouge_2 = eval_rouge_2 / iter_batch
    avg_rouge_l = eval_rouge_l / iter_batch

    avg_meteor = eval_meteor / iter_batch
    avg_cider = eval_cider / iter_batch

    logger.info(f"  avg_bleu_1: {avg_bleu_1}, avg_bleu_2: {avg_bleu_2}, avg_bleu_3: {avg_bleu_3}, avg_bleu_4: {avg_bleu_4}")
    logger.info(f"  avg_rouge_1: {avg_rouge_1}, avg_rouge_2: {avg_rouge_2}, avg_rouge_l: {avg_rouge_l}")
    logger.info(f"  avg_meteor: {avg_meteor}, avg_cider: {avg_cider}")

    #### add evaluation here #################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,                               
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--llm_type",
        type=str,
        default='llama_1',
        help=(
            "choose the LLM backbone to train model  "
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='/home/user/Downloads/2005047/MEIT/ECG_LLMs/cache-dir/',
        help=(
            "path to save model"
        ),
    )
    parser.add_argument(
        "--ecg_model_type",
        type=str,
        default="ResNet50",
        help=(
            'ResNet18, ResNet34, ResNet50, ResNet101, ResNet152'
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='/home/wan.512/ECG_LLMs/ECG_gen/instruct_data/mimic_ecg.jsonl',
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/mmlu/llama-7B/"
    )

    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )

    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )

   
    
    

    parser.add_argument(
        "--train_or_test",
        type=str,
        default="train",
        help=(
            "train or test the ECG_LLMs model "
        ),
    )

    parser.add_argument(
        "--ecg_model_ckpt_path",
        type=str,
        default=None,
        help=(
            "checkpoint of ecg_model  "
        ),
    )

    parser.add_argument(
        "--ecg_layer_idxs",
        type=str,
        default="all",
        help=(
            "31,30,29,28,27"
        ),
    )

    parser.add_argument(
        "--dev_ratio",
        type=float,
        default=0.1,
        help="Proportion of the dataset to include in the development set, should be between 0.0 and 1.0.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        ),
    ) 
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=2,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--lora_model_ckpt_path",
        type=str,
        default='/home/user/Downloads/2005047/MEIT/ECG_LLMs/save_dir_llama1',
        help=(
            "checkpoint of lora model  "
        ),
    )


    args = parser.parse_args()

    main(args)