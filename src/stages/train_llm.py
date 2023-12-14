"""
This script is responsible for training the autoregressive llm.

Usage: script_name.py params_parent_field(input) train_dataset_filename(input) ft_output_path(output)
"""

import argparse
import time
from dotenv import load_dotenv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, AutoConfig, PretrainedConfig, EarlyStoppingCallback, EvalPrediction
import torch
from peft import LoraConfig
from datasets import load_from_disk
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import yaml
import wandb
from sacrebleu import corpus_bleu
from tqdm import tqdm

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('params_parent_field')
parser.add_argument('train_dataset_filename')
parser.add_argument('ft_output_path')

args = parser.parse_args()

# Access the arguments
params_parent_field = args.params_parent_field
train_dataset_filename = args.train_dataset_filename
finetuned_path = args.ft_output_path

# parse params
with open("params.yaml", 'r') as file:
    params = yaml.safe_load(file)[params_parent_field]

##load hugging-face token
load_dotenv()
hf_token = os.environ["HF_ACCESS_TOKEN"]
hf_token_write = os.environ["HF_ACCESS_TOKEN_WRITE"]

##get wandb variables
wandb_entity= params["wandb_params"]["entity"]
wandb_project = params["wandb_params"]["project"]

## Hyperparams
hf_hub_repo = params["hf_hub_repo"]
model_id = params["model_params"]["model_id"]
pre_train_config = params["model_params"]["pre_train_config"]
post_train_config = params["model_params"]["post_train_config"]
tokenizer_id = params["tokenizer_params"]["tokenizer_id"]
max_seq_length = params["tokenizer_params"]["max_seq_length"]
max_new_tokens = params["tokenizer_params"]["max_new_tokens"]
ft_dataset_filename = train_dataset_filename
train_batch_size = params["training_config"]["train_batch_size"]
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

val_batch_size = params["validation_config"]["val_batch_size"]
validate_every_n_steps = params["validation_config"]["validate_every_n_steps"]
early_stopping_patience = params["validation_config"]["early_stopping_patience"]

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
tokenizer.pad_token = "</p>" #define padding token
tokenizer.padding_side = "right" #define padding side


#load config 
model_config:PretrainedConfig = AutoConfig.from_pretrained(model_id)

print("original model config: ")
print(model_config)

#alter config for training
if pre_train_config:
    for key, value in pre_train_config.items():
        setattr(model_config, key, value)

#load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map=device_map,
    token=hf_token,
    config=model_config
)
print("model config used for training: ")
print(model_config)

#load train dataset
full_dataset = load_from_disk(ft_dataset_filename, keep_in_memory=True)

train_split = full_dataset.filter(lambda x: x['split'] == 'train')#get training split
val_split = full_dataset.filter(lambda x: x['split'] == "val") #get validation split
test_split = full_dataset.filter(lambda x: x['split'] == "test") #get test split

#sanity check
assert len(full_dataset) == len(train_split) + len(val_split) + len(test_split), f"Something went wrong during the splitting process of the dataset... All the splits together have a length of {len(train_split) + len(val_split) + len(test_split)} but it has to sum up to {len(full_dataset)}"

## Start training
train_args = TrainingArguments(
    output_dir=finetuned_path + "/train-out",
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=val_batch_size,
    gradient_accumulation_steps=grad_accumulation_steps,
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    optim=optimizer,
    logging_steps=logging_steps,
    num_train_epochs=num_train_epochs,
    lr_scheduler_type=lr_scheduler_type,
    group_by_length=True, #try to put same length inputs into the same batch
    gradient_checkpointing=gradient_checkpointing,
    report_to="wandb", #MLOps plattform
    evaluation_strategy="steps", #validate every n steps (eval_steps)
    save_strategy="steps", #checkpointing strategy must be the same as evaluation_strategy
    eval_steps=validate_every_n_steps,
    load_best_model_at_end=True, #if true -> the model with the lowest validation loss will be loaded after training
    metric_for_best_model="loss", #use validation loss as metric for best model 
    greater_is_better=False #we use loss as metric for best model -> lowest loss is better
)

def formatting_func(dataset, eval=False):
    output_texts = []
    for question, context, answer in zip(dataset[dataset_columns['question']], dataset[dataset_columns['context']], dataset[dataset_columns['answer']]):
        prompt = f"[INST] Nachfolgend bekommst du eine Frage gestellt mit dem best passenden Kontext. Versuche Frage mithilfe des Kontextes zu beantworten. [/INST]\n\n [FRAGE] {question} [/FRAGE]\n\n [KONTEXT] {context} [/KONTEXT]\n\n ANTWORT:\n"
        
        prompt += f"{answer}{tokenizer.eos_token}" if not eval else "" #add answer if not in eval mode
        
        output_texts.append(prompt)
    
    return output_texts

def generate_predictions(model, tokenizer, test_split):
    model.eval() # Setzt das Modell in den Evaluierungsmodus

    predictions = []
    labels = []
    for i, prompt in enumerate(tqdm(formatting_func(test_split, eval=True))):

        #tokenize and make prediction
        inputs = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True, max_length=max_seq_length-max_new_tokens).input_ids
        decoded_preds = tokenizer.decode(model.generate(input_ids=inputs.to(model.device), max_new_tokens=max_new_tokens)[0], skip_special_tokens=True)

        predictions.append(decoded_preds)
        labels.append(test_split[i][dataset_columns['answer']])

    return predictions, labels

def test_stage(trainer: SFTTrainer):
    test_predictions, labels = generate_predictions(trainer.model, trainer.tokenizer, test_split)
    bleu_score = corpus_bleu(test_predictions, [labels]).score

    trainer.log({"test_bleu": bleu_score})

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_split,
    eval_dataset=val_split,
    formatting_func=formatting_func,
    max_seq_length=max_seq_length,
    args=train_args,
    peft_config=lora_config,
    neftune_noise_alpha=neftune_noise_alpha,
    data_collator = DataCollatorForCompletionOnlyLM("ANTWORT:\n", tokenizer=tokenizer) if train_on_completion_only else None #train on completion only (text after "ANTWORT:\n") if train_on_completion_only == True
)

# add EarlyStoppingCallback
trainer.add_callback(EarlyStoppingCallback(
    early_stopping_patience=early_stopping_patience
))

#manually init wandb run and store name
wandb_run_name = wandb.init(entity=wandb_entity, project=wandb_project).name

#start training
trainer.train()

##test stage
test_stage(trainer)

##model saving 
model_config:PretrainedConfig = trainer.model.config #get config

#alter config for saving
if post_train_config:
    for key, value in post_train_config.items():
        setattr(model_config, key, value)

## Save model, tokenizer and model config locally
trainer.model.save_pretrained(finetuned_path)
model_config.save_pretrained(finetuned_path)
trainer.tokenizer.save_pretrained(finetuned_path)

def create_commit_message(component:str):
    return f"Update of {component}\n\nWandB run name: ({wandb_run_name})"

#push model, config and tokenizer to hub if required
if hf_hub_repo != None:
    trainer.model.push_to_hub(hf_hub_repo, token=hf_token_write, commit_message=create_commit_message("LoRA adapters"))
    model_config.push_to_hub(hf_hub_repo, token=hf_token_write, commit_message=create_commit_message("model config"))
    trainer.tokenizer.push_to_hub(hf_hub_repo, token=hf_token_write, commit_message=create_commit_message("tokenizer"))

#wait a sec to avoid simulateous access to files
time.sleep(1)