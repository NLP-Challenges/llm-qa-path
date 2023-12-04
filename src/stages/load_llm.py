"""
This script is responsible for the inference of an llm.

Usage: script_name.py params_parent_field(input) finetuned_path(input) context_path(input) question(input) 
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, LlamaTokenizer, AutoConfig
from dotenv import load_dotenv
import os
from peft import LoraConfig, PeftModel
import torch
import argparse
import yaml
import json

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('params_parent_field')
parser.add_argument('finetuned_path')
parser.add_argument('context_path')
parser.add_argument('question')

args = parser.parse_args()

# Access the arguments
params_parent_field = args.params_parent_field
finetuned_path = args.finetuned_path
context_path = args.context_path
question = args.question

# parse params
with open("params.yaml", 'r') as file:
    params = yaml.safe_load(file)[params_parent_field]

# params
device_map = params["model_params"]["device_map"]

max_new_tokens = params["generation_params"]["max_new_tokens"]
top_k = params["generation_params"]["top_k"]
penalty_alpha = params["generation_params"]["penalty_alpha"]
temperature = params["generation_params"]["temperature"]

max_seq_length = params["tokenizer_params"]["max_seq_length"]


#load hugging-face token
load_dotenv()
hf_token = os.environ["HF_ACCESS_TOKEN"]

#read context
with open(context_path, "r", encoding="utf-8") as f:
    context = json.dumps(json.load(f), ensure_ascii=False)

#load lora config
lora_config = LoraConfig.from_pretrained(finetuned_path) 

#load model config
model_config = AutoConfig.from_pretrained(finetuned_path)

#define bnb config
bnb_config = BitsAndBytesConfig(
    **{key:eval(value) if isinstance(value, str) and "torch." in value else value for (key, value) in  params["BitsAndBytesConfig"].items()}
)

#load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    lora_config.base_model_name_or_path, 
    quantization_config=bnb_config, 
    device_map=device_map,
    config=model_config
)
model = PeftModel.from_pretrained(model, finetuned_path) #attach lora layers

tokenizer = AutoTokenizer.from_pretrained(finetuned_path)

def predict(model:PreTrainedModel, tokenizer:LlamaTokenizer, question:str, context:str):
    model.eval()
    
    prompt = f"[INST] Nachfolgend bekommst du eine Frage gestellt mit dem best passenden Kontext. Versuche Frage mithilfe des Kontextes zu beantworten. [/INST]\n\n [FRAGE] {question} [/FRAGE]\n\n [KONTEXT] {context} [/KONTEXT]\n\n ANTWORT:\n"

    inputs = tokenizer(
        prompt, 
        truncation=True,
        padding=False,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    
    outputs = model.generate(
        input_ids=inputs["input_ids"].to(model.device),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        
        #Contrastive search: https://huggingface.co/blog/introducing-csearch
        penalty_alpha=penalty_alpha, 
        top_k=top_k
    )

    return tokenizer.decode(outputs[:, inputs["input_ids"].shape[1]:][0], skip_special_tokens=True)

print("Frage: ", question)
print("Antwort: ", predict(model, tokenizer, question, context))