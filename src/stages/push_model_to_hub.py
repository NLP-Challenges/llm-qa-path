
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import LoraConfig, PeftModel
from dotenv import load_dotenv
import yaml
import os
import torch

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('params_parent_field')
parser.add_argument('finetuned_path')
parser.add_argument("repo_name")

args = parser.parse_args()

# Access the arguments
params_parent_field = args.params_parent_field
finetuned_path = args.finetuned_path
repo_name = args.repo_name

# parse params
with open("params.yaml", 'r') as file:
    params = yaml.safe_load(file)[params_parent_field]

# params
device_map = params["model_params"]["device_map"]


#load hugging-face token
load_dotenv()
hf_token_write = os.environ["HF_ACCESS_TOKEN_WRITE"]

#define bnb config
bnb_config = BitsAndBytesConfig(
    **{key:eval(value) if isinstance(value, str) and "torch." in value else value for (key, value) in  params["BitsAndBytesConfig"].items()}
)

#load lora config
lora_config = LoraConfig.from_pretrained(finetuned_path) 

#load model config
model_config = AutoConfig.from_pretrained(finetuned_path)

#load model
model = AutoModelForCausalLM.from_pretrained(
    lora_config.base_model_name_or_path, 
    quantization_config=bnb_config, 
    device_map=device_map,
    config=model_config
)
model = PeftModel.from_pretrained(model, finetuned_path) #attach lora layers

#load tokenizer
tokenizer = AutoTokenizer.from_pretrained(finetuned_path)

#push model, config and tokenizer to hub
model.push_to_hub(repo_name, token=hf_token_write)
model.config.push_to_hub(repo_name, token=hf_token_write)
tokenizer.push_to_hub(repo_name, token=hf_token_write)