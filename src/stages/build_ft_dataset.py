"""
This script is responsible for creating the llm fine-tuning dataset. The germanquad dataset (https://huggingface.co/datasets/deepset/germanquad) is currently used.
To allow the recognition of wrong context, the context was wapped for a couple of questions and the answer replaced.

Usage: script_name.py params_parent_field(input) dataset_filename(output)
"""

import argparse
from datasets import load_dataset
import pandas as pd
import datasets
import time
import numpy as np
import yaml

np.random.seed(1234) 

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

def sample(df_original:pd.DataFrame, df_swapped:pd.DataFrame, size:int, fraw_swapped:float):
    n_original = int(size * (1-fraw_swapped)) #calculate number of samples from original dataframe
    n_swapped = size - n_original #calculate number of samples from swapped dataframe

    #sample from dataframes
    split_original = df_original.sample(n=n_original, random_state=1234)
    split_swapped = df_swapped.sample(n=n_swapped, random_state=1234)

    #combine
    split = pd.concat([split_original, split_swapped], ignore_index=True)

    #get unique contexts
    used_context = split.context.unique()

    #drop all observations with this used context from original and swapped dataframe
    df_original.drop(df_original[df_original.context.isin(used_context)].index, inplace=True)
    df_swapped.drop(df_swapped[df_swapped.context.isin(used_context)].index, inplace=True)

    return split #return split

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('params_parent_field')
parser.add_argument('dataset_filename')

args = parser.parse_args()

# Access the arguments
params_parent_field = args.params_parent_field
dataset_filename = args.dataset_filename

# parse params
with open("params.yaml", 'r') as file:
    params = yaml.safe_load(file)[params_parent_field]

# concatenate training and test split
df_original:pd.DataFrame = pd.concat(
    [pd.DataFrame(load_dataset("deepset/germanquad")["train"]), pd.DataFrame(load_dataset("deepset/germanquad")["test"])],
    ignore_index=True,
)

#remove additional context in context field like "Recht_der_Vereinigten_Staaten\n\n=== Amerikanisches Common Law ===\n'Actual context'"
pattern = r'^.*?\n+\s*?={1,}.*?={1,}\s*?\n+' #pattern to search for
df_original = df_original[df_original.context.str.contains(pattern)] #only use observations containing this pattern
df_original["context"] = df_original.context.str.replace(pattern, '', regex=True).str.replace("\n", " ") #remove those patterns

#extract text from answer filed
df_original["answers"] = df_original.answers.apply(lambda x: x["text"][0])

#remove id filed
df_original.drop(columns="id", inplace=True) 

#reset index
df_original.reset_index(drop=True, inplace=True)

#add new column for "can be answered"
df_original["can_be_answered"] = [True]*len(df_original)

print(f"total number of observations in original dataset is {len(df_original)}")
print(f"total number of unique context is {len(df_original.context.unique())}. To be safe keep the total length of all splits together lower than this number!")

#swap context
df_swapped = swap_context(df_original)

## sampling
train_split = sample(df_original, df_swapped, params["n_train"], params["frac_swapped"])
val_split = sample(df_original, df_swapped, params["n_val"], params["frac_swapped"])
test_split = sample(df_original, df_swapped, params["n_test"], params["frac_swapped"])

#add new column which identifies the split
train_split["split"] = ["train"]*len(train_split)
val_split["split"] = ["val"]*len(val_split)
test_split["split"] = ["test"]*len(test_split)

if len(np.intersect1d(train_split.context.unique(), val_split.context.unique())) > 0:
    raise Exception("Leakage between train_split and val_split")

if len(np.intersect1d(train_split.context.unique(), test_split.context.unique())) > 0:
    raise Exception("Leakage between train_split and test_split")

if len(np.intersect1d(val_split.context.unique(), test_split.context.unique())) > 0:
    raise Exception("Leakage between val_split and test_split")

final_df = pd.concat([train_split, val_split, test_split]).sample(frac = 1, random_state=1234).reset_index(drop=True, inplace=False)

print("final dataset length: ", len(final_df))

datasets.Dataset.from_pandas(final_df).save_to_disk(dataset_filename)

#wait a sec to avoid simulateous access to files
time.sleep(1)

