"""
Dieses Script generiert ein Embedder, welcher für das generieren von Embeddings verwendet werden kann.

Usage: script_name.py embedder_filename(output)
"""

import argparse
from dill import dump
from langchain.embeddings import HuggingFaceEmbeddings

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('embedder_filename')

args = parser.parse_args()

# Access the arguments
embedder_filename = args.embedder_filename

#https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b
embedder = HuggingFaceEmbeddings(model_name="msmarco-distilbert-base-tas-b")

with open(embedder_filename, "wb") as f:
    dump(embedder, f)
