"""
This script is responsible for creating the llm fine-tuning dataset. The germanquad dataset (https://huggingface.co/datasets/deepset/germanquad) is currently used.
To allow the recognition of wrong context, the context was wapped for a couple of questions and the answer replaced.

Usage: script_name.py params_parent_field(input) train_dataset_filename(output)
"""

import argparse
from datasets import load_dataset
import pandas as pd
import datasets
import time
import numpy as np
import yaml

def swap_context(df:pd.DataFrame):
    """ Copy df and swap the context randomly but also change answer to 'Leider liegen mir dazu keine Informationen vor'
    Args:
        df (pd.DataFrame)
    Returns:
        dataframe with swapped context
    """
    def shuffle(input):
        copy = input.copy()

        while True:
            np.random.shuffle(copy)

            if np.all(copy != input):
                break

        return copy
    
    #get all the questions 
    grouped = df.groupby("context")

    #extract all the context texts
    keys_from = list(grouped.groups.keys())
    keys_to = shuffle(keys_from) #shuffle

    #mapper
    mapper = {key: value for key, value in zip(keys_from, keys_to)}

    def swap(key, df):
        #set new context from mapper and replace answers
        df = df.copy()
        df["context"] = [mapper[key]]*len(df)
        df["answers"] = ["Leider liegen mir dazu keine Informationen vor"]*len(df)
        df["can_be_answered"] = [False]*len(df)

        return df

    return pd.concat([swap(key, df) for (key, df) in grouped], ignore_index=True)

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('params_parent_field')
parser.add_argument('train_dataset_filename')

args = parser.parse_args()

# Access the arguments
params_parent_field = args.params_parent_field
train_dataset_filename = args.train_dataset_filename

# parse params
with open("params.yaml", 'r') as file:
    params = yaml.safe_load(file)[params_parent_field]

#set numpy seed (for swap_context)
np.random.seed(params["seed"]) 

#load germansquad dataset (train split) and convert to pandas dataframe
df = pd.DataFrame(load_dataset("deepset/germanquad", split="train"))

#remove additional context in context field like "Recht_der_Vereinigten_Staaten\n\n=== Amerikanisches Common Law ===\n'Actual context'"
pattern = r'^.*?\n+\s*?={1,}.*?={1,}\s*?\n+' #pattern to search for
filtered_df = df[df.context.str.contains(pattern)].copy() #only use observations containing this pattern
filtered_df["context"] = filtered_df.context.str.replace(pattern, '', regex=True).str.replace("\n", " ") #remove those patterns

#extract text from answer filed
filtered_df["answers"] = filtered_df.answers.apply(lambda x: x["text"][0])

#remove id filed
filtered_df.drop(columns="id", inplace=True) 

#reset index
filtered_df.reset_index(drop=True, inplace=True)

#add new column for "can be answered"
filtered_df["can_be_answered"] = [True]*len(filtered_df)

print("total number of question, context, answer pairs available: ", len(filtered_df))

#swap context
swapped = swap_context(filtered_df)

df_frac_original = filtered_df.sample(n=params["n_original"], frac=params["frac_original"], random_state=params["seed"])
df_frac_swapped = swapped.sample(n=params["n_swapped"], frac=params["frac_swapped"], random_state=params["seed"])

print("number of question, context, answer pairs: ", len(df_frac_original))
print("number of non fitting question, context, answer pairs: ", len(df_frac_swapped))

#concat shortened original and shortened swapped.
final_df = pd.concat([df_frac_original, df_frac_swapped], ignore_index=True).sample(frac = 1, random_state=params["seed"]).reset_index(drop=True, inplace=False)

print("final dataset length: ", len(final_df))

datasets.Dataset.from_pandas(final_df, split="train").save_to_disk(train_dataset_filename)

#wait a sec to avoid simulateous access to files
time.sleep(1)

