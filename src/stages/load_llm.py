"""
This script is responsible for the inference of an llm.

Usage: script_name.py params_parent_field(input) finetuned_path(input) question(input) context(input)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, LlamaTokenizer
from dotenv import load_dotenv
import os
from peft import LoraConfig, PeftModel
import torch
import argparse
import yaml

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('params_parent_field')
parser.add_argument('finetuned_path')
parser.add_argument('question')
parser.add_argument('context')

args = parser.parse_args()

# Access the arguments
params_parent_field = args.params_parent_field
finetuned_path = args.finetuned_path
question = args.question
context = args.context

# parse params
with open("params.yaml", 'r') as file:
    params = yaml.safe_load(file)[params_parent_field]

#load hugging-face token
load_dotenv()
hf_token = os.environ["HF_ACCESS_TOKEN"]

#load lora config from model
lora_config = LoraConfig.from_pretrained(finetuned_path + "/model") 

#define bnb config
bnb_config = BitsAndBytesConfig(
    **{key:eval(value) if isinstance(value, str) and "torch." in value else value for (key, value) in  params["BitsAndBytesConfig"].items()}
)

#load model and tokenizer
ft_model = AutoModelForCausalLM.from_pretrained(
    lora_config.base_model_name_or_path, 
    quantization_config=bnb_config, 
    device_map=params["model_params"]["device_map"]
)
ft_model = PeftModel.from_pretrained(ft_model, finetuned_path + "/model") #attach lora layers

ft_tokenizer = AutoTokenizer.from_pretrained(finetuned_path + "/tokenizer")

def predict(model:PreTrainedModel, tokenizer:LlamaTokenizer, question:str, context:str):
    model.eval()
    
    prompt = (
        "Nachfolgend ist eine Frage gestellt mit dem entsprechenden Kontext.\n"
        "Schreibe eine passende Antwort zur Frage und beziehe den Kontext mit hinein.\n"
        "Die Antwort soll eine angemessene Länge besitzen.\n\n"
        "### Frage:\n"
        f"{question}\n\n"
        "### Kontext:\n"
        f"{context}\n\n"
        "### Antwort:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids=inputs["input_ids"].to(ft_model.device),
        max_new_tokens=params["model_params"]["max_new_tokens"], 
        temperature=params["model_params"]["temperature"],
        do_sample=True,
    )

    return tokenizer.decode(outputs[:, inputs["input_ids"].shape[1]:][0], skip_special_tokens=True)

print("Frage: ", question)
print("Antwort: ", predict(ft_model, ft_tokenizer, question, context))