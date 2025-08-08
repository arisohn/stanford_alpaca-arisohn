#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
import transformers
import utils
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from datasets import Dataset
from skeleton import _print, _pprint, CustomSFTTrainer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{response}"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(SFTConfig):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


"""
class CustomSFTTrainer(SFTTrainer):
    def training_step(self, *args, **kwargs):
        rank = int(os.environ['RANK'])
        step = self.state.global_step

        if rank == 0 and step == 0:
            inputs = args[1] if len(args) > 1 else kwargs.get('inputs', None)

            input_ids = inputs.get('input_ids', None)
            labels = inputs.get('labels', None)
            attention_mask = inputs.get('attention_mask', None)
            print("\n\n\n-----------------------------------------------------------------------")
            print("[CustomSFTTrainer] input_id:", input_ids[0])
            print("[CustomSFTTrainer] label:", labels[0])
            print("[CustomSFTTrainer] attention_mask:", attention_mask[0])
            
            decoded = self.processing_class.convert_ids_to_tokens(input_ids[0])
            print("[CustomSFTTrainer] input_id decoded:", decoded)
            decoded = self.processing_class.decode(input_ids[0], skip_special_tokens=False)
            print("[CustomSFTTrainer] input_id decoded:", decoded)

            label = torch.where(labels[0] == -100, torch.tensor(self.processing_class.pad_token_id, device=labels.device), labels[0])
            decoded = self.processing_class.convert_ids_to_tokens(label)
            print("[CustomSFTTrainer] label decoded:", decoded)
            decoded = self.processing_class.decode(label, skip_special_tokens=False)
            print("[CustomSFTTrainer] label decoded:", decoded)
            print("\n\n\n")

        return super().training_step(*args, **kwargs)
"""


def formatting_prompts_func(examples):
        output_texts = []

        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples["input"][i]
            response = examples["output"][i]

            if len(input_text) >= 2:
                prompt_input = PROMPT_DICT['prompt_input']
                prompt_input = prompt_input.format_map({
                    "instruction": instruction,
                    "input": input_text,
                    "response": response
                })

                output_texts.append(prompt_input)
            else:
                prompt_no_input = PROMPT_DICT['prompt_no_input']
                prompt_no_input = prompt_no_input.format_map({
                    "instruction": instruction,
                    "response": response 
                })

                output_texts.append(prompt_no_input)

        return output_texts


def load_dataset_from_json(data_path: str) -> Dataset:
    """Load dataset from JSON file and convert to HuggingFace Dataset."""
    logging.warning("Loading data...")
    list_data_dict = utils.jload(data_path)
    
    # Convert list of dicts to dict of lists for Dataset
    dataset_dict = {
        "instruction": [d.get("instruction", "") for d in list_data_dict],
        "input": [d.get("input", "") for d in list_data_dict],
        "output": [d.get("output", "") for d in list_data_dict]
    }
    
    return Dataset.from_dict(dataset_dict)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    _print("-" * 100)
    _print(model_args)
    _print(data_args)
    _print(training_args)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    train_dataset = load_dataset_from_json(data_args.data_path)
    
    # Create data collator for completion only
    # https://huggingface.co/docs/trl/v0.9.6/sft_trainer : to handle LLAMA1 case
    response_template_with_context = "\n### Response:\n"  # We added context here: "\n". This is enough for this tokenizer
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids,
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Configure SFTTrainer
    """
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        formatting_func=lambda x: formatting_prompts_func(x, tokenizer),
        data_collator=data_collator,
    )
    """

    trainer = CustomSFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,
    )
    
    trainer.train()
    #trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir) # FSDP 때문에, 여기서 저장할때 에러가 발생하여 모델에 저장된 값이 이상할때가 있음 -> requirements_trl_fsdp_workaround.txt 를 사용하여 패키지 설치!


if __name__ == "__main__":
    train()

