#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
finetune_philosophy_llm.py

This script demonstrates how to fine-tune a language model (LLM) on a philosophy-related dataset.
It includes functions for:
    - Logging into HuggingFace Hub
    - Loading the base model and tokenizer
    - Generating sample text (pre fine-tuning)
    - Preparing and tokenizing the dataset
    - Fine-tuning the model with LoRA
    - Merging and saving the fine-tuned model
    - Running inference with the fine-tuned model

Best practices such as modularity, clear docstrings, and inline comments have been applied.
"""

import os
import torch
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def login_hf(token: str) -> None:
    """
    Logs into the HuggingFace Hub using the provided token.

    Args:
        token (str): The HuggingFace API token.
    """
    login(token)
    print("Logged in to HuggingFace Hub.")


def load_model(model_name: str, device_map: str = "auto") -> (AutoTokenizer, AutoModelForCausalLM):
    """
    Loads the tokenizer and model from the given model name.

    Args:
        model_name (str): The identifier of the base model.
        device_map (str): Device mapping strategy; default is "auto".

    Returns:
        tuple: A tuple containing the tokenizer and the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set pad token to EOS token to ensure proper padding behavior
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )
    # Load and update generation configuration
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    print(f"Loaded model and tokenizer from {model_name}.")
    return tokenizer, model

def build_input_text(example: dict) -> str:
    """
    Builds the input text for a dataset example.

    Args:
        example (dict): A dictionary containing the dataset sample with keys 'text' and 'category'.

    Returns:
        str: A formatted string to be used as input.
    """
    return (
        "Instruction:\n"
        "You are given a philosophical statement. Identify the school of philosophy it represents.\n\n"
        "Input:\n"
        f"{example['text']}\n\n"
        "Output:\n"
        f"{example['category']}\n"
    )


def tokenize_function(example: dict, tokenizer, max_length: int = 4096) -> dict:
    """
    Tokenizes a single example from the dataset.

    Args:
        example (dict): A single dataset example.
        tokenizer: The tokenizer.
        max_length (int): Maximum length for tokenization.

    Returns:
        dict: The tokenized output.
    """
    full_text = build_input_text(example)
    return tokenizer(
        full_text,
        padding=True,
        truncation=True,
        max_length=max_length
    )


def prepare_datasets(tokenizer, datasetpath) -> (any, any):
    """
    Loads and tokenizes the dataset for training and evaluation.

    Args:
        tokenizer: The tokenizer used for tokenization.
        dataset_name (str): The identifier of the dataset from HuggingFace datasets.

    Returns:
        tuple: A tuple containing the tokenized training and evaluation datasets.
    """
    # Load streaming dataset
    dataset = load_dataset("csv",data_files=datasetpath)

    # Shuffle the dataset
    dataset = dataset.shuffle(42)

    # Split the dataset into training and evaluation sets
    split_dataset = dataset['train'].train_test_split(test_size=0.1)

    # Convert each split into a regular Hugging Face Dataset
    train_dataset = split_dataset["train"]
    eval_dataset  = split_dataset["test"]


    # Tokenize each dataset example
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=False)
    eval_dataset = eval_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=False)

    print("Datasets loaded and tokenized.")
    return train_dataset, eval_dataset


def fine_tune_model(model, tokenizer, train_dataset, eval_dataset, output_dir: str = "finetuned_model") -> None:
    """
    Fine-tunes the provided model using the specified datasets and saves the fine-tuned model.

    Args:
        model: The base language model.
        tokenizer: The tokenizer.
        train_dataset: The tokenized training dataset.
        eval_dataset: The tokenized evaluation dataset.
        output_dir (str): Directory where the fine-tuned model will be saved.
    """
    # Configure LoRA parameters for fine-tuning
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # Wrap the base model with LoRA
    model = get_peft_model(model, lora_config)
    print("Model wrapped with LoRA configuration.")

    # Set up the data collator for language modeling (without MLM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        logging_steps=50,
        learning_rate=1e-5,
        warmup_steps=100,
        gradient_accumulation_steps=4,
        fp16=False,
        bf16=True if torch.cuda.is_bf16_supported() else False,
        report_to="none"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    print("Starting fine-tuning...")
    trainer.train()

    # Merge LoRA weights and save the final model and tokenizer
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}.")

def main():
    """
    Main function to execute the fine-tuning and inference pipeline.
    """
    # Define parameters
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")

    BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B"
    OUTPUT_DIR = "finetuned-philosophers-llama3.2-1b"
    DATASET_NAME = "stanford_philosophy_first_10000.csv"
    # Log in to HuggingFace Hub
    login_hf(HF_TOKEN)

    # Load the base model and tokenizer
    tokenizer, model = load_model(BASE_MODEL_NAME)

    print("Start generating the dataset:")
    # Prepare the datasets for fine-tuning
    train_dataset, eval_dataset = prepare_datasets(tokenizer,DATASET_NAME)

    print("Start fine-tuning the model:")
    # Fine-tune the model using the prepared datasets
    fine_tune_model(model, tokenizer, train_dataset, eval_dataset, output_dir=OUTPUT_DIR)

if __name__ == "__main__":
    main()
