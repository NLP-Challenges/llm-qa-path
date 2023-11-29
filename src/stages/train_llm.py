"""
This script is responsible for training the autoregressive llm.

Usage: script_name.py params_parent_field(input) train_dataset_filename(input) ft_output_path(output)
"""

import argparse
import time
from dotenv import load_dotenv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, AutoConfig
import torch
from peft import LoraConfig
from datasets import load_from_disk
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
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
model_id = params["model_params"]["model_id"]
override_model_config = params["model_params"]["override_model_config"]
tokenizer_id = params["tokenizer_params"]["tokenizer_id"]
max_seq_length = params["tokenizer_params"]["max_seq_length"]
finetuned_path = ft_output_path
ft_dataset_filename = train_dataset_filename
train_batch_size = params["training_config"]["batch_size"]
grad_accumulation_steps = params["training_config"]["grad_accumulation_steps"]
optimizer = params["training_config"]["optimizer"]
learning_rate = params["training_config"]["learning_rate"]
num_train_epochs = params["training_config"]["num_train_epochs"]
warmup_steps = params["training_config"]["warmup_steps"]
logging_steps = params["training_config"]["logging_steps"]
lr_scheduler_type = params["training_config"]["lr_scheduler_type"]
device_map = params["training_config"]["device_map"]
train_on_completion_only = params["training_config"]["completion_only"]
gradient_checkpointing = params["training_config"]["gradient_checkpointing"]
neftune_noise_alpha = params["training_config"]["neftune_noise_alpha"]
dataset_columns = params["DatasetColumns"]

## Configuration
lora_config = LoraConfig(
    **params["LoraConfig"]
)

bnb_config = BitsAndBytesConfig(
    **{key:eval(value) if isinstance(value, str) and "torch." in value else value for (key, value) in  params["BitsAndBytesConfig"].items()}
)

## Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_id,
    token=hf_token,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


#load config if override required
model_config = None #default none
if override_model_config:
    model_config = AutoConfig.from_pretrained(override_model_config)

#load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map=device_map,
    token=hf_token,
    config=model_config
)
model.config.use_cache = False #turn of cache to prevent problems during training!


## Load dataset
def formatter(example):
    #it is required to have a space between "[/KONTEXT]\n\n" and "ANTWORT:\n" because the DataCollatorForCompletionOnlyLM doesn't find the pattern otherwise
    prompt = f"[INST] Nachfolgend bekommst du eine Frage gestellt mit dem best passenden Kontext. Versuche Frage mithilfe des Kontextes zu beantworten. [/INST]\n\n [FRAGE] {example[dataset_columns['question']]} [/FRAGE]\n\n [KONTEXT] {example[dataset_columns['context']]} [/KONTEXT]\n\n ANTWORT:\n{example[dataset_columns['answer']]}{tokenizer.eos_token}"

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
    warmup_steps=warmup_steps,
    optim=optimizer,
    logging_steps=logging_steps,
    num_train_epochs=num_train_epochs,
    lr_scheduler_type=lr_scheduler_type,
    group_by_length=True,
    gradient_checkpointing=gradient_checkpointing
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=train_args,
    peft_config=lora_config,
    neftune_noise_alpha=neftune_noise_alpha,
    data_collator = DataCollatorForCompletionOnlyLM("ANTWORT:\n", tokenizer=tokenizer) if train_on_completion_only else None #train on completion only (text after "ANTWORT:\n") if train_on_completion_only == True
)
trainer.train()

trainer.model.config.use_cache = True #turn caching back on again

## Save model, tokenizer and model config
trainer.model.save_pretrained(finetuned_path)
trainer.model.config.save_pretrained(finetuned_path)
trainer.tokenizer.save_pretrained(finetuned_path)

#wait a sec to avoid simulateous access to files
time.sleep(1)