"""
Dieses Script is für das erstellen einer Chroma Vektordatenbank verantwortlich.
Auf dieser Datenbank kann eine Similarity-Query durchgeführt werden.

Usage: script_name.py corpus_filename(input) embedder_filename(input) vector_database_filename(output)
"""

from langchain.vectorstores import Chroma
import argparse
from dill import load
import time

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

#load corpus
with open(corpus_filename, 'rb') as f:
    corpus = load(f)

#load embedder
with open(embedder_filename, 'rb') as f:
    embedder = load(f)

#create vectorstore and save it to vector_database_filename
db = Chroma.from_texts(corpus, embedder, persist_directory=vector_database_filename)
db.persist()

#disconnect from chroma
del db

#wait a sec to avoid simulateous access to files
time.sleep(1)