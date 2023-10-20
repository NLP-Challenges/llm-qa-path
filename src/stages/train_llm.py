"""
Dieses Script is für das Training eines llms verantwortlich.

Usage: script_name.py params_parent_field(input) train_dataset_filename(input) ft_output_path(output)
"""

import argparse
import time
from dotenv import load_dotenv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from datasets import load_from_disk
from trl import SFTTrainer
import yaml

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('params_parent_field')
parser.add_argument('train_dataset_filename')
parser.add_argument('ft_output_path')
args = parser.parse_args()

# Access the arguments
params_parent_field = args.params_parent_field
train_dataset_filename = args.train_dataset_filename
ft_output_path = args.ft_output_path

# parse params
with open("params.yaml", 'r') as file:
    params = yaml.safe_load(file)[params_parent_field]

##load hugging-face token
load_dotenv()
hf_token = os.environ["HF_ACCESS_TOKEN"]

##set wandb environment variables
os.environ["WANDB_ENTITY"] = params["wandb_params"]["entity"]
os.environ["WANDB_PROJECT"] = params["wandb_params"]["project"]

## Hyperparams
base_model_id = params["model_params"]["base_model_id"]
finetuned_path = ft_output_path
ft_dataset_filename = train_dataset_filename
train_set_text_field = "text"
train_batch_size = params["training_config"]["batch_size"]
grad_accumulation_steps = params["training_config"]["grad_accumulation_steps"]
optimizer = params["training_config"]["optimizer"]
learning_rate = params["training_config"]["learning_rate"]
max_steps = params["training_config"]["max_steps"]
max_seq_length = params["model_params"]["max_seq_length"]
device_map = params["training_config"]["device_map"]

## Configuration
lora_config = LoraConfig(
    **params["LoraConfig"]
)

bnb_config = BitsAndBytesConfig(
    **{key:eval(value) if isinstance(value, str) and "torch." in value else value for (key, value) in  params["BitsAndBytesConfig"].items()}
)

## Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    token=hf_token,
)
tokenizer.pad_token = "</p>"
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map=device_map,
    token=hf_token,
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

base_model:AutoPeftModelForCausalLM = prepare_model_for_kbit_training(base_model)
base_model = get_peft_model(base_model, lora_config) # add lora adapters
print(base_model.print_trainable_parameters())

## Load dataset
def formatter(example):
    prompt = (
        "Nachfolgend ist eine Frage gestellt mit dem entsprechenden Kontext sowie der passenden Antwort.\n"
        "Schreibe eine passende Antwort zur Frage und beziehe den Kontext mit hinein.\n"
        "Die Antwort soll so kurz wie möglich sein.\n\n"
        "### Frage:\n"
        f"{example['question']}\n\n"
        "### Kontext:\n"
        f"{example['context']}\n\n"
        "### Antwort:\n"
        f"{example['answers']}{tokenizer.eos_token}"
    )

    return {"text": prompt}

#load train dataset
train_dataset = load_from_disk(ft_dataset_filename, keep_in_memory=True)

#add text column
train_dataset = train_dataset.map(formatter)

## Start training
train_args = TrainingArguments(
    output_dir=finetuned_path + "/train-out",
    per_device_train_batch_size=train_batch_size,
    gradient_accumulation_steps=grad_accumulation_steps,
    learning_rate=learning_rate,
    optim=optimizer,
    logging_steps=1,
    max_steps=max_steps,
    lr_scheduler_type="constant",
    group_by_length=True
)

fine_tuning = SFTTrainer(
    model=base_model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field=train_set_text_field,
    max_seq_length=max_seq_length,
    args=train_args
)

fine_tuning.train()

## Save model and tokenizer
fine_tuning.model.save_pretrained(finetuned_path + "/model")
fine_tuning.tokenizer.save_pretrained(finetuned_path + "/tokenizer")

#wait a sec to avoid simulateous access to files
time.sleep(1)