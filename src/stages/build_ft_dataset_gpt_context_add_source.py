import argparse
import time
from datasets import load_from_disk
from dotenv import load_dotenv
import datasets
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

import pandas as pd
import numpy as np
import json

def get_chunks(series: pd.Series, max_chunks: int = 4):
    """
    Funktion zum Aufteilen von Texten in der Series in Chunks.
    Jeder Text wird zuerst in Sätze aufgeteilt und dann in Chunks unterteilt.
    
    :param series: pandas Series mit Texten.
    :param max_chunks: Maximale Anzahl von Chunks, in die ein Text aufgeteilt werden kann.
    :return: Eine neue Series, wobei jeder Eintrag eine Liste von Chunks ist.
    """
    
    def split_text(text:str, num_chunks:int):
        return [" ".join(splits) for splits in np.array_split([sentence+"." for sentence in text.split(". ")], num_chunks)] #get chunks
    
    chunksize = np.random.choice(list(range(1, max_chunks+1)), size=len(series))
    
    return [split_text(text, chunksize[i]) for i, text in enumerate(series)]

def get_source(text:str):
    template = (
        """Stelle dir eine passende Quelle vor für den folgenden Text. Bitte gib nur den Namen der Quelle an, ohne zusätzliche Informationen wie Kapitel, Seitenzahl oder detaillierte Beschreibung.
        """
    )

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{message}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    messages = chat_prompt.format_prompt(message=text).to_messages()

    return chat(messages).content

def gpt_formatter(series: pd.Series):
    formatted = []
    for chunks in tqdm(series):
        formatted.append(json.dumps([{"QUELLE": get_source(text), "INHALT": text} for text in chunks], ensure_ascii=False))

    return formatted

parser = argparse.ArgumentParser()

# Add positional arguments
parser.add_argument('train_dataset_filename_input')
parser.add_argument('train_dataset_filename_output')

args = parser.parse_args()

# Access the arguments
train_dataset_filename_input = args.train_dataset_filename_input
train_dataset_filename_output = args.train_dataset_filename_output

load_dotenv()

# load germansquad dataset (train split) and convert to pandas dataframe
df = pd.DataFrame(load_from_disk(train_dataset_filename_input, keep_in_memory=True))

# Setup OpenAI API
chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", request_timeout=60)

df["context"] = get_chunks(df.context)
df["context"] = gpt_formatter(df.context)

# save updated dataset
datasets.Dataset.from_pandas(df, split="train").save_to_disk(train_dataset_filename_output)

#wait a sec to avoid simulateous access to files
time.sleep(1)