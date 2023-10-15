"""
Dieses Script ist für das erstellen des llm fine-tuning datasets verantwortlich. Zum jetzigen Zeitpunkt wird das germanquad datenset (https://huggingface.co/datasets/deepset/germanquad) verwendet.
Usage: script_name.py train_dataset_filename(output)
"""

import argparse
from datasets import load_dataset
import pandas as pd
import datasets

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('dataset_filename')

args = parser.parse_args()

# Access the arguments
dataset_filename = args.dataset_filename


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

datasets.Dataset.from_pandas(filtered_df, split="train").save_to_disk(dataset_filename)

