"""
This script is responsible for creating the llm fine-tuning dataset. The germanquad dataset (https://huggingface.co/datasets/deepset/germanquad) is currently used.
To allow the recognition of wrong context, the context was wapped for a couple of questions and the answer replaced.

Usage: script_name.py seed(input) train_dataset_filename(output)
"""

import argparse
from datasets import load_dataset
import pandas as pd
import datasets
import time
import numpy as np

def swap_context(df:pd.DataFrame):
    """ Copy df and swap the context randomly but also change answer
    """
    def shuffle(input):
        copy = input.copy()

        while True:
            np.random.shuffle(copy)

            if np.all(copy != input):
                break

        return copy
        
    grouped = df.groupby("context")

    keys_from = list(grouped.groups.keys())
    keys_to = shuffle(keys_from)

    mapper = {key: value for key, value in zip(keys_from, keys_to)}

    def swap(key, df):
        df = df.copy()
        df["context"] = [mapper[key]]*len(df)
        df["answers"] = ["Leider liegen mir dazu keine Informationen vor"]*len(df)

        return df

    return pd.concat([swap(key, df) for (key, df) in grouped], ignore_index=True)

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('seed')
parser.add_argument('train_dataset_filename')

args = parser.parse_args()

# Access the arguments
seed_ = int(args.seed)
train_dataset_filename = args.train_dataset_filename

np.random.seed(seed_) #set numpy seed

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

#swap context
swapped = swap_context(filtered_df)

#concat shortened swapped and stortened filteted df (can be deleted later) / shuffle
final_df = pd.concat([filtered_df.sample(frac=0.7), swapped.sample(frac=0.3)], ignore_index=True).sample(frac = 1)

print(final_df)

datasets.Dataset.from_pandas(final_df, split="train").save_to_disk(train_dataset_filename)

#wait a sec to avoid simulateous access to files
time.sleep(1)

