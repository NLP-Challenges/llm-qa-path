"""
Dieses Script is für die Inferenz eines llms verantwortlich.

Usage: script_name.py finetuned_path(input) question(input) context(input)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, LlamaTokenizer
from dotenv import load_dotenv
import os
from peft import LoraConfig, PeftModel
import torch
import argparse

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('finetuned_path')
parser.add_argument('question')
parser.add_argument('context')

args = parser.parse_args()

# Access the arguments
finetuned_path = args.finetuned_path
question = args.question
context = args.context


#load hugging-face token
load_dotenv()
hf_token = os.environ["HF_ACCESS_TOKEN"]

#load lora config from model
lora_config = LoraConfig.from_pretrained(finetuned_path + "/model") 

#define bnb config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

#load model and tokenizer
ft_model = AutoModelForCausalLM.from_pretrained(
    lora_config.base_model_name_or_path, 
    quantization_config=bnb_config, 
    device_map="cuda:0"
)
ft_model = PeftModel.from_pretrained(ft_model, finetuned_path + "/model") #attach lora layers

ft_tokenizer = AutoTokenizer.from_pretrained(finetuned_path + "/tokenizer")

def predict(model:PreTrainedModel, tokenizer:LlamaTokenizer, question:str, context:str):
    model.eval()
    
    prompt = (
        "Nachfolgend ist eine Frage gestellt mit dem entsprechenden Kontext\n"
        "Schreibe eine passende Antwort als vollständiger Satz zur Frage und beziehe den Kontext mit hinein\n\n"
        "### Frage:\n"
        f"{question}\n\n"
        "### Kontext:\n"
        f"{context}\n\n"
        "### Antwort:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda:0"), attention_mask=inputs["attention_mask"], max_new_tokens=200, pad_token_id=tokenizer.pad_token_id, do_sample=True, temperature=1.0)

    return tokenizer.decode(outputs[:, inputs["input_ids"].shape[1]:][0], skip_special_tokens=True)

print("Frage: ", question)
print("Antwort: ", predict(ft_model, ft_tokenizer, question, context))