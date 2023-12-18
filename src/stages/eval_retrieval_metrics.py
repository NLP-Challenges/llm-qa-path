"""
Generates metrics for evaluating the retrieval

Usage: script_name.py vector_database_filename(input) embedder_filename(input) questions_file(input) metrics_file(output)
"""

from langchain.vectorstores import Chroma
from dill import load
import argparse
import tempfile
import os
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('vector_database_filename')
parser.add_argument('embedder_filename')
parser.add_argument('questions_file')
parser.add_argument('metrics_file')

args = parser.parse_args()

# Access the arguments
vector_database_filename = args.vector_database_filename
embedder_filename = args.embedder_filename
questions_file = args.questions_file
metrics_file = args.metrics_file

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

# load embedder
with open(embedder_filename, 'rb') as f:
    embedder = load(f)

# make a copy of the chroma folder
temp = copy_to_temp_folder(vector_database_filename)

# load chroma db from temporary folder
vectorstore = Chroma(persist_directory=temp, embedding_function=embedder)
num_docs = len(vectorstore.get()["ids"])

#read parquet file
df = pd.read_parquet(questions_file)

#iterate over questions
ranks = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    block_id, question = row.loc["id"], row.loc["question"] #get block id and question which was generated to that block

    #execute query on that question and get ids ordered by similarity
    block_ids_search = vectorstore._collection.query(
        query_embeddings=vectorstore._embedding_function.embed_query(question),
        n_results=num_docs,
    )["ids"][0]

    #calculate rank
    block_rank = np.argwhere(np.array(block_ids_search) == block_id).flatten()[0] + 1

    ranks.append(block_rank)

mrr = np.mean(1 / np.array(ranks)) #calculate the mean reciprocal rank

print("ranks: ", ranks)

#try to delete folder with db
delete_folder(temp)

#write json to output
with open(metrics_file, "w") as f:
    f.write(json.dumps({"mrr":mrr})) 