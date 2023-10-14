"""
Dieses Script generiert ein Embedder, welcher für das generieren von Embeddings verwendet werden kann.

Usage: script_name.py model_name(input) embedder_filename(output)
"""

import argparse
from dill import dump
from langchain.embeddings import HuggingFaceEmbeddings
import time

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('model_name')
parser.add_argument('embedder_filename')

args = parser.parse_args()

# Access the arguments
model_name = args.model_name
embedder_filename = args.embedder_filename

embedder = HuggingFaceEmbeddings(model_name=model_name)

with open(embedder_filename, "wb") as f:
    dump(embedder, f)

#wait a sec to avoid simulateous access to files
time.sleep(1)
