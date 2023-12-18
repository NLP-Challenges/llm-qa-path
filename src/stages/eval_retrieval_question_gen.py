"""
Generates questions for retrieval evaluation.

Usage: script_name.py vector_database_filename(input) embedder_filename(input) questions_file(output)
"""

from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import argparse
import tempfile
import os
import shutil
from dill import load
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('vector_database_filename')
parser.add_argument('embedder_filename')
parser.add_argument('questions_file')

args = parser.parse_args()

# Access the arguments
vector_database_filename = args.vector_database_filename
embedder_filename = args.embedder_filename
questions_file = args.questions_file

##openai token
load_dotenv()

#######################configuration#######################
model_name = "LeoLM/leo-mistral-hessianai-7b-chat"
device = "cuda:0"
max_new_tokens = 50

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

sys_instruction = "Deine Aufgabe ist es EINE (PASSENDE) Frage zum user Input zu erstellen. Antworte so kurz wie möglich!"
###########################################################

def copy_to_temp_folder(source_folder:str):
    # create a temporary folder
    temp_folder = tempfile.mkdtemp()

    # copy files to temporary folder
    for item in os.listdir(source_folder):
        s = os.path.join(source_folder, item)
        d = os.path.join(temp_folder, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

    return temp_folder

def delete_folder(folder:str):
    try:
        # try delete folder
        shutil.rmtree(folder)

    except Exception as ex: 
        print(f"Temporaray folder could not be deleted: {ex}. The folder has to be deleted manually!")    

def get_prompt(text:str):
    prompt = f"<|im_start|>system\n{sys_instruction}<|im_end|>\n"
    
    prompt += f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

    return prompt  

#load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map=device,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def generate_answer(text:str):
    prompt = get_prompt(text)

    inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    #get all newly generated tokens
    generated_tokens = output[0][inputs["input_ids"].shape[1]:]

    #check if last token is actually end of sentence token
    if tokenizer.decode(generated_tokens[[-1]]) == "</s>":
        return tokenizer.decode(generated_tokens, skip_special_tokens=True, encoding="utf-8").strip()
    
    #if text not generated to end -> return nan
    else:
        return pd.NA

# load embedder
with open(embedder_filename, 'rb') as f:
    embedder = load(f)

# make a copy of the chroma folder
temp = copy_to_temp_folder(vector_database_filename)

# load chroma db from temporary folder
vectorstore = Chroma(persist_directory=temp, embedding_function=embedder)

#iterate over blocks
ids = []
questions = []
for id in tqdm(vectorstore.get()["ids"]):
    #generate text from block text
    text = vectorstore.get(id)["documents"][0]
    question = generate_answer(text)

    #append to list
    ids.append(id)
    questions.append(question)

df = pd.DataFrame(
    {
        "id": ids,
        "question": questions
    }
)

df = df.dropna(inplace=False) #drop all rows with na in it (those questions where not generated to end)

#store data
df.to_parquet(questions_file)
