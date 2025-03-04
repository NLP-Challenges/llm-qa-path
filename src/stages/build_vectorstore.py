"""
This script is responsible for creating a Chroma vector database.
A similarity query can be carried out on this database.

Usage: script_name.py corpus_filename(input) embedder_filename(input) vector_database_filename(output)
"""

from typing import Iterable
from langchain.vectorstores import Chroma
import argparse
from dill import load
import time
from utils.data_helpers import load_docs_from_jsonl

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('corpus_filename')
parser.add_argument('embedder_filename')
parser.add_argument('vector_database_filename')

args = parser.parse_args()

# Access the arguments
corpus_filename = args.corpus_filename
embedder_filename = args.embedder_filename
vector_database_filename = args.vector_database_filename

# load corpus
corpus = load_docs_from_jsonl(corpus_filename)

# load embedder
with open(embedder_filename, 'rb') as f:
    embedder = load(f)

# create vectorstore and save it to vector_database_filename
db = Chroma.from_documents(corpus, embedder, persist_directory=vector_database_filename)

vectorstore = Chroma("langchain_store", embedder)

db.persist()

# disconnect from chroma
del db

# wait a sec to avoid simulateous access to files
time.sleep(1)