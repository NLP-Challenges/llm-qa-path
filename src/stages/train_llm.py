import argparse
from datasets import load_from_disk
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from dotenv import load_dotenv
import os

#load hugging-face token
load_dotenv()
hf_token = os.environ["HUGGINGFACE_TOKEN"]

#set wandb environment variables
os.environ["WANDB_ENTITY"] = "t_buess"
os.environ["WANDB_PROJECT"] = "chatbot-qa"

#load arguments
parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('ft_dataset_filename')

args = parser.parse_args()

# Access the arguments
model_name = "meta-llama/Llama-2-7b-chat-hf"
ft_dataset_filename = args.ft_dataset_filename
max_seq_length = 4096

def formatter(example):
    prompt = (
        "Nachfolgend ist eine Frage gestellt mit dem entsprechenden Kontext sowie der passenden Antwort",
        "Schreibe eine passende Antwort zur Frage und beziehe den Kontext mit hinein",
        "### Frage:\n",
        f"{example['question']}\n\n",
        "### Kontext:\n",
        f"{example['context']}\n\n",
        "### Antwort:\n",
        f"{example['answers']}",
    )

    return {"text": prompt}

#load train dataset
train_dataset = load_from_disk(ft_dataset_filename)

#add text column
train_dataset = train_dataset.map(formatter)

#according to https://medium.com/@ud.chandra/instruction-fine-tuning-llama-2-with-pefts-qlora-method-d6a801ebb19 & https://www.youtube.com/watch?v=eTieetk2dSw

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda:0",
    use_auth_token=hf_token
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = "[PAD]"

training_args = TrainingArguments(
    output_dir="data/processed/training-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=500,
    report_to="wandb",
    fp16=True
)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()