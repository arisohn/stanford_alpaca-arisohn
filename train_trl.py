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

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
import transformers
import utils
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from datasets import Dataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
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


def formatting_prompts_func(examples: Dict[str, List], tokenizer: transformers.PreTrainedTokenizer) -> List[str]:
    """Format the dataset for SFTTrainer."""
    # Handle both single example and batched examples
    if isinstance(examples["instruction"], str):
        # Single example case
        instruction = examples["instruction"]
        input_text = examples.get("input", "")
        output = examples["output"]
        
        if input_text:
            prompt = PROMPT_DICT["prompt_input"].format_map({
                "instruction": instruction,
                "input": input_text
            })
        else:
            prompt = PROMPT_DICT["prompt_no_input"].format_map({
                "instruction": instruction
            })
        
        # Tokenize prompt to check if we have room for output
        prompt_tokens = tokenizer(prompt, truncation=False, return_tensors=None)
        prompt_length = len(prompt_tokens["input_ids"])
        
        # Reserve space for output and EOS token
        max_output_length = tokenizer.model_max_length - prompt_length - 1
        
        if max_output_length > 50:  # Ensure minimum output space
            # Truncate output if needed
            output_tokens = tokenizer(output, truncation=False, return_tensors=None)
            if len(output_tokens["input_ids"]) > max_output_length:
                # Truncate output to fit
                truncated_output_ids = output_tokens["input_ids"][:max_output_length]
                output = tokenizer.decode(truncated_output_ids, skip_special_tokens=True)
            
            text = prompt + output + tokenizer.eos_token
            return text
        else:
            logging.warning(f"Skipping example - prompt too long ({prompt_length} tokens)")
            # Return minimal valid example to avoid error
            return "### Instruction:\nHello\n\n### Response:\nHi" + tokenizer.eos_token
    else:
        # Batched examples case
        output_texts = []
        prompt_input = PROMPT_DICT["prompt_input"]
        prompt_no_input = PROMPT_DICT["prompt_no_input"]
        
        for i in range(len(examples["instruction"])):
            # Handle missing or empty input field
            if "input" in examples and i < len(examples["input"]):
                input_text = examples["input"][i] if examples["input"][i] else ""
            else:
                input_text = ""
            
            if input_text:
                prompt = prompt_input.format_map({
                    "instruction": examples["instruction"][i],
                    "input": input_text
                })
            else:
                prompt = prompt_no_input.format_map({
                    "instruction": examples["instruction"][i]
                })
            
            # Tokenize prompt to check if we have room for output
            prompt_tokens = tokenizer(prompt, truncation=False, return_tensors=None)
            prompt_length = len(prompt_tokens["input_ids"])
            
            # Reserve space for output and EOS token
            max_output_length = tokenizer.model_max_length - prompt_length - 1
            
            if max_output_length > 50:  # Ensure minimum output space
                output = examples["output"][i]
                # Truncate output if needed
                output_tokens = tokenizer(output, truncation=False, return_tensors=None)
                if len(output_tokens["input_ids"]) > max_output_length:
                    # Truncate output to fit
                    truncated_output_ids = output_tokens["input_ids"][:max_output_length]
                    output = tokenizer.decode(truncated_output_ids, skip_special_tokens=True)
                
                text = prompt + output + tokenizer.eos_token
                output_texts.append(text)
            else:
                logging.warning(f"Skipping example {i} - prompt too long ({prompt_length} tokens)")
        
        # Always return at least one valid example to avoid empty batch error
        if not output_texts:
            output_texts.append("### Instruction:\nHello\n\n### Response:\nHi" + tokenizer.eos_token)
        
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

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() and training_args.fp16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    # Set special tokens
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

    # Load dataset
    train_dataset = load_dataset_from_json(data_args.data_path)
    
    # Create data collator for completion only
    '''
    response_template = "### Response:\n"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    '''
    
    # https://huggingface.co/docs/trl/v0.9.6/sft_trainer
	response_template_with_context = "\n### Response:\n"  # We added context here: "\n". This is enough for this tokenizer
	response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]

	data_collator = DataCollatorForCompletionOnlyLM(
		response_template_ids,
		tokenizer=tokenizer,
		mlm=False
	)
       
    # Configure SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        formatting_func=lambda x: formatting_prompts_func(x, tokenizer),
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
