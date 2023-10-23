"""
This script is responsible for loading a Chroma vector database.
A similarity query can be carried out on this database.

Usage: script_name.py vector_database_filename(input) embedder_filename(input) query(input)
"""

from langchain.vectorstores import Chroma
import argparse
from dill import load
import time
import shutil
import tempfile
import os
import gc

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('vector_database_filename')
parser.add_argument('embedder_filename')
parser.add_argument('query')

args = parser.parse_args()

# Access the arguments
vector_database_filename = args.vector_database_filename
embedder_filename = args.embedder_filename
query = args.query

def copy_folder(source_folder:str):
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

def replace_original(source_folder:str, temp_folder:str):
    # delete original folder
    shutil.rmtree(source_folder)

    #move temporary folder to original location
    shutil.move(temp_folder, source_folder)

#load embedder
with open(embedder_filename, 'rb') as f:
    embedder = load(f)

#make a copy of the chroma folder
temp = copy_folder(vector_database_filename)

# load chroma db
db = Chroma(persist_directory=vector_database_filename, embedding_function=embedder)

# query
docs = db.similarity_search(query)
print(docs[0].page_content)

#disconnect from chroma
del db
gc.collect()

#wait a sec to avoid simulateous access to files
time.sleep(1)

#replace original chroma folder
replace_original(vector_database_filename, temp)

#wait a sec to avoid simulateous access to files
time.sleep(1)