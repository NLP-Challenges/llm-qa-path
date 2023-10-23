"""
This script is responsible for dividing text into text blocks (corpus).

Usage: script_name.py text_filename(input) curpus_filename(output)
"""

import argparse
from dill import dump
import time

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('text_filename')
parser.add_argument('corpus_filename')

#add optional arguments
#parser.add_argument('--extra', '-e')

args = parser.parse_args()

# Access the arguments
text_filename = args.text_filename
corpus_filename = args.corpus_filename
#extra = args.extra

#read file content
with open(text_filename, "r") as f:
    text = f.read()

#split by empty lines
corpus = text.split("\n\n")

#remove first block (just source description)
corpus = corpus[1:]

with open(corpus_filename, "wb") as f:
    dump(corpus, f)

#wait a sec to avoid simulateous access to files
time.sleep(1)


