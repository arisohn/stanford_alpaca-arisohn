import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    TrainerCallback
)
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import logging
from transformers import DataCollatorForLanguageModeling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_model_and_tokenizer(model_name):
    logger.info(f"Loading model: {model_name}")
    
    # For demo, using regular loading without quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def prepare_dataset(dataset_name="tatsu-lab/alpaca", tokenizer=None):
    logger.info("Loading dataset")
    dataset = load_dataset(dataset_name, split="train[:50]")  # Very small subset for quick demo
    
    # Remove unnecessary columns
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["instruction", "input", "output"]])
    
    # Format and tokenize the dataset
    def format_and_tokenize(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples["input"][i] if examples["input"][i] else ""
            output = examples["output"][i]
            
            if input_text:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}{tokenizer.eos_token}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}{tokenizer.eos_token}"
            texts.append(text)
        
        # Tokenize the texts
        tokenized = tokenizer(texts, truncation=True, padding=False, max_length=512)
        return tokenized
    
    # Apply formatting and tokenization
    dataset = dataset.map(format_and_tokenize, batched=True, remove_columns=dataset.column_names)
    
    # Add a text field temporarily for logging, then remove it after
    def add_text_field(examples):
        texts = []
        for i in range(len(examples["input_ids"])):
            text = tokenizer.decode(examples["input_ids"][i], skip_special_tokens=False)
            texts.append(text)
        return {"text": texts}
    
    dataset_with_text = dataset.map(add_text_field, batched=True)
    
    return dataset, dataset_with_text

def main():
    # Using smaller model for faster demo
    model_name = "gpt2"  
    output_dir = "./sft_results_demo"
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(model_name)
    
    # Prepare dataset with tokenization
    dataset, dataset_with_text = prepare_dataset(tokenizer=tokenizer)
    
    # Create data collator that only calculates loss on completion
    response_template = "### Response:"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Custom SFTTrainer with logging
    class LoggingSFTTrainer(SFTTrainer):
        def __init__(self, *args, log_freq=1, **kwargs):
            super().__init__(*args, **kwargs)
            self.log_freq = log_freq
            self.step_count = 0
        
        def compute_loss(self, model, inputs, **kwargs):
            # Log inputs and labels
            if self.step_count % self.log_freq == 0:
                print(f"{'='*80}")
                print(f"== Training Step {self.step_count}:")
                print(f"{'='*80}")

                input_ids = inputs.get('input_ids')
                labels = inputs.get('labels')
                
                if input_ids is not None:
                    # Log each example in the batch
                    batch_size = input_ids.shape[0]
                    for i in range(min(batch_size, 2)):  # Show max 2 examples per batch
                        print(f"== Example {i}:")
                        
                        # Decode and show the text
                        decoded_text = tokenizer.decode(input_ids[i], skip_special_tokens=False)
                        print(f"== Decoded text: {decoded_text[:300]}...")
                        print(f"== Input IDs shape: {input_ids[i].shape}")
                        print(f"== First 50 token IDs: {input_ids[i][:50].tolist()}")
                        
                        # Show labels if they exist
                        if labels is not None:
                            # Count non-padding tokens in labels
                            non_padding = (labels[i] != -100).sum().item()
                            print(f"== Labels shape: {labels[i].shape}")
                            print(f"== Non-padding tokens in labels: {non_padding}")
                            print(f"== First 50 label IDs: {labels[i][:50].tolist()}")
                            
                            # Show which parts are masked (label = -100)
                            masked_positions = (labels[i] == -100).sum().item()
                            print(f"== Masked positions (-100): {masked_positions}")
                            
                            # Show the actual target tokens (non-masked labels)
                            target_tokens = labels[i][labels[i] != -100]
                            if len(target_tokens) > 0:
                                decoded_targets = tokenizer.decode(target_tokens[:20], skip_special_tokens=False)
                                print(f"== First 20 target tokens (decoded): {decoded_targets}")
                                
                            # Log how DataCollatorForCompletionOnlyLM works:
                            # It sets labels to -100 for all tokens before "### Response:\n"
                            # This means loss is only calculated on the response part
                
                print(f"{'='*80}\n\n\n")
            
            self.step_count += 1
            
            # Call parent compute_loss
            return super().compute_loss(model, inputs, **kwargs)
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=8,  # Smaller rank for demo
        lora_alpha=16,
        target_modules=["c_attn"],  # GPT-2 attention layers
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Training arguments
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        optim="adamw_torch",  # Standard optimizer
        save_steps=25,
        logging_steps=5,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=torch.cuda.is_available(),  # Use fp16 only if GPU available
        bf16=False,
        max_grad_norm=0.3,
        max_steps=10,  # Very limited steps for quick demo
        warmup_ratio=0.1,
        group_by_length=True,
        lr_scheduler_type="linear",
        report_to="none",
        remove_unused_columns=False,
        seed=42,
    )
    
    # Print sample formatted data before training
    logger.info("\n" + "="*80)
    logger.info("Sample formatted training data:")
    logger.info("="*80)
    for i in range(min(3, len(dataset_with_text))):
        logger.info(f"\nSample {i}:")
        text = dataset_with_text[i].get("text", "")
        logger.info(text[:300] + "..." if len(text) > 300 else text)
    logger.info("="*80 + "\n")
    
    # Create trainer with logging
    # Using pre-tokenized dataset with DataCollatorForCompletionOnlyLM
    trainer = LoggingSFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_arguments,
        train_dataset=dataset,
        peft_config=peft_config,
        data_collator=data_collator
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    
    logger.info("Training completed!")
    
    # Test inference
    test_inference(model, tokenizer)

def test_inference(model, tokenizer):
    logger.info("Testing inference with trained model")
    
    test_prompts = [
        "### Instruction:\nWhat is Python?\n\n### Response:\n",
        "### Instruction:\nExplain machine learning in simple terms.\n\n### Response:\n",
        "### Instruction:\nWrite a short poem about coding.\n\n### Response:\n"
    ]
    
    device = next(model.parameters()).device
    
    model.eval()
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt[:50]}...")
        print(f"Response: {response[len(prompt):]}")

if __name__ == "__main__":
    main()
