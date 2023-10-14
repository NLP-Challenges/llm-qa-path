"""
Dieses Script is für das Laden einer Chroma Vektordatenbank verantwortlich.
Auf dieser Datenbank kann eine Similarity-Query durchgeführt werden.

Usage: script_name.py vector_database_filename(input) embedder_filename(input) query(input)
"""

from langchain.vectorstores import Chroma
import argparse
from dill import load
import time

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

#load embedder
with open(embedder_filename, 'rb') as f:
    embedder = load(f)

# load chroma db
db = Chroma(persist_directory=vector_database_filename, embedding_function=embedder)

# query
docs = db.similarity_search(query)
print(docs[0].page_content)

#disconnect from chroma
del db

#wait a sec to avoid simulateous access to files
time.sleep(1)